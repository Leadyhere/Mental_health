"""Phase transition orchestrator. `handle_client_message` is the single
entry point chat_ws.py calls: it mutates the given SessionState in place
and returns the list of ServerMessage to send back.
"""
from __future__ import annotations

from app.conversation import narrative_phase, question_selector
from app.conversation.emergency_gate import evaluate_emergency
from app.conversation.narrative_analysis import analyze_narrative
from app.conversation.narrative_phase import NarrativeAction
from app.conversation.question_bank import QUESTION_BANK
from app.crisis_resources import CRISIS_RESOURCES, EMERGENCY_MESSAGE
from app.llm.classifier import classify_risk
from app.llm.safety_filter import apply_confidence_escalation, filter_outbound_message
from app.models.schemas import (
    ClientMessage,
    ConversationPhase,
    RiskAssessment,
    ServerMessage,
    StructuredAnswer,
    SubLabels,
    Utterance,
)


def _extract_text(message: ClientMessage) -> str:
    if message.type == "chip_select":
        return str(message.payload.get("chip_value", ""))
    return str(message.payload.get("text", ""))


def _next_turn_id(state) -> int:
    return len(state.utterances) + 1


def _build_session_context(state) -> str:
    lines = ["--- Narrative ---"]
    for u in state.utterances:
        if u.role == "user":
            lines.append(u.text)
    if state.narrative_analysis:
        lines.append("\n--- Narrative analysis ---")
        lines.append(f"emotions: {state.narrative_analysis.emotions}")
        lines.append(f"coping_signals: {state.narrative_analysis.coping_signals}")
        lines.append(f"support_signals: {state.narrative_analysis.support_signals}")
        lines.append(f"functional_impact: {state.narrative_analysis.functional_impact}")
        lines.append(f"risk_language_flags: {state.narrative_analysis.risk_language_flags}")
    lines.append("\n--- Structured answers ---")
    for a in state.structured_answers:
        lines.append(f"{a.question_id}: {a.chip_selected or a.answer_text}")
    return "\n".join(lines)


async def _trigger_emergency(state) -> list[ServerMessage]:
    state.phase = ConversationPhase.emergency
    state.emergency_triggered = True
    return [
        ServerMessage(type="phase_change", payload={"phase": "emergency"}),
        ServerMessage(type="crisis_banner", payload={"message": EMERGENCY_MESSAGE, "resources": CRISIS_RESOURCES}),
    ]


async def _ask_question(state, question) -> ServerMessage:
    state.pending_question_id = question.id
    return ServerMessage(
        type="chip_options",
        payload={
            "question_id": question.id,
            "prompt": question.prompt,
            "options": question.chip_options,
            "allow_free_text": True,
            "asked_verbatim": question.asked_verbatim,
        },
    )


async def _finalize_session(state) -> list[ServerMessage]:
    session_context = _build_session_context(state)
    assessment, llm_says_emergency = await classify_risk(session_context)
    assessment = apply_confidence_escalation(assessment)

    # Escalation can itself push the level to Emergency (e.g. an unparseable/
    # failed LLM call falls back to Severe at 0.0 confidence, which then
    # escalates one level) -- that must route through the same hard gate as
    # an explicit LLM/hardcoded Emergency signal, never just get reported.
    if (
        llm_says_emergency
        or assessment.risk_level == "Emergency"
        or evaluate_emergency(state.structured_answers, llm_says_emergency=llm_says_emergency)
    ):
        return await _trigger_emergency(state)

    state.current_risk_assessment = assessment
    state.phase = ConversationPhase.closing

    from app.services.logging_service import log_session_decision
    await log_session_decision(state, assessment)

    return [
        ServerMessage(type="phase_change", payload={"phase": "closing"}),
        ServerMessage(type="report_ready", payload={"session_id": str(state.session_id)}),
    ]


async def handle_client_message(state, message: ClientMessage) -> list[ServerMessage]:
    if state.phase in (ConversationPhase.closing, ConversationPhase.emergency):
        return []

    user_text = _extract_text(message)
    state.utterances.append(Utterance(turn_id=_next_turn_id(state), role="user", text=user_text))

    if state.phase == ConversationPhase.narrative:
        num_narrative_turns = sum(1 for u in state.utterances if u.role == "user")
        result = await narrative_phase.process_narrative_turn(
            user_text,
            has_prior_substantive_turn=num_narrative_turns > 1,
            completion_check_already_asked=state.narrative_completion_check_asked,
        )

        if result.action == NarrativeAction.emergency:
            return await _trigger_emergency(state)

        if result.action == NarrativeAction.ask_completion_check:
            state.narrative_completion_check_asked = True
            state.utterances.append(Utterance(turn_id=_next_turn_id(state), role="assistant", text=result.assistant_text))
            return [ServerMessage(type="assistant_text", payload={"text": result.assistant_text})]

        if result.action == NarrativeAction.ack:
            state.utterances.append(Utterance(turn_id=_next_turn_id(state), role="assistant", text=result.assistant_text))
            return [ServerMessage(type="assistant_text", payload={"text": result.assistant_text})]

        # move_to_analysis
        narrative_text = "\n".join(u.text for u in state.utterances if u.role == "user")
        analysis, _inhouse_signal = await analyze_narrative(narrative_text, turns=[u.text for u in state.utterances if u.role == "user"])
        state.narrative_analysis = analysis
        state.phase = ConversationPhase.structured

        from app.llm.client import complete
        from app.llm.prompts.loader import render
        transition_text = render("transition_to_structured")

        next_question = question_selector.select_next_question(
            state.narrative_analysis, state.structured_answers, state.structured_question_count
        )
        messages = [
            ServerMessage(type="phase_change", payload={"phase": "structured"}),
            ServerMessage(type="assistant_text", payload={"text": transition_text}),
        ]
        if next_question is None:
            messages.extend(await _finalize_session(state))
        else:
            messages.append(await _ask_question(state, next_question))
        return messages

    if state.phase == ConversationPhase.structured:
        question = next((q for q in QUESTION_BANK if q.id == state.pending_question_id), None)
        if question is not None:
            answer = StructuredAnswer(
                question_id=question.id,
                question_text=question.prompt,
                answer_text=user_text if message.type != "chip_select" else None,
                chip_selected=user_text if message.type == "chip_select" else None,
                asked_verbatim=question.asked_verbatim,
            )
            state.structured_answers.append(answer)
            state.structured_question_count += 1
            state.pending_question_id = None

            if evaluate_emergency(state.structured_answers):
                return await _trigger_emergency(state)

        next_question = question_selector.select_next_question(
            state.narrative_analysis, state.structured_answers, state.structured_question_count
        )
        if next_question is None:
            return await _finalize_session(state)
        return [await _ask_question(state, next_question)]

    return []
