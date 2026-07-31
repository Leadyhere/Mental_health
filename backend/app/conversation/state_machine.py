"""Phase transition orchestrator. `handle_client_message` is the single
entry point chat_ws.py calls: it mutates the given SessionState in place
and returns the list of ServerMessage to send back.
"""
from __future__ import annotations

from app.conversation import narrative_phase, question_selector
from app.conversation.emergency_gate import evaluate_emergency
from app.conversation.narrative_phase import NarrativeAction
from app.conversation.question_bank import QUESTION_BANK
from app.crisis_resources import CRISIS_RESOURCES, EMERGENCY_MESSAGE
from app.llm.classifier import classify_risk
from app.llm.client import complete_json
from app.llm.prompts.loader import render
from app.llm.safety_filter import apply_confidence_escalation, filter_outbound_message
from app.models.schemas import (
    ClientMessage,
    ConversationPhase,
    NarrativeAnalysis,
    RiskAssessment,
    ServerMessage,
    StructuredAnswer,
    SubLabels,
    Utterance,
)
from app.nlp_engine import NLPEngine

_EMPTY_ANSWERED = {
    "emotional_clarification": False,
    "coping_behaviors": False,
    "support_person": False,
    "distress_intensity": False,
    "feels_safe": False,
    "meaning_readiness": False,
    "open_to_support": False,
}


async def _analyze_narrative(narrative_text: str, turns: list[str]) -> tuple[NarrativeAnalysis, dict | None]:
    system = render("narrative_analysis", narrative_text=narrative_text)
    try:
        data = await complete_json(system=system, user="Analyze this narrative.")
        analysis = NarrativeAnalysis(
            key_events=data.get("key_events", []),
            timeline=data.get("timeline"),
            emotions=data.get("emotions", []),
            coping_signals=data.get("coping_signals", []),
            support_signals=data.get("support_signals", []),
            functional_impact=data.get("functional_impact"),
            risk_language_flags=data.get("risk_language_flags", []),
            answered_questions=data.get("answered_questions", dict(_EMPTY_ANSWERED)),
        )
        NLPEngine.log_teacher_pair(narrative_text, data)
        return analysis, None
    except Exception:
        analysis = NarrativeAnalysis(answered_questions=dict(_EMPTY_ANSWERED))
        return analysis, None


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

    is_explicit_done = (message.type == "done_sharing") or bool(message.payload.get("done_sharing"))
    user_text = _extract_text(message)
    if user_text:
        state.utterances.append(Utterance(turn_id=_next_turn_id(state), role="user", text=user_text))

    # 2-Tier Crisis Pre-check across all phases
    if user_text:
        is_crisis, _rationale = NLPEngine.check_crisis_override(user_text)
        if is_crisis:
            return await _trigger_emergency(state)

    if state.phase == ConversationPhase.narrative:
        num_narrative_turns = sum(1 for u in state.utterances if u.role == "user")
        result = await narrative_phase.process_narrative_turn(
            user_text,
            has_prior_substantive_turn=num_narrative_turns > 1,
            completion_check_already_asked=state.narrative_completion_check_asked,
            explicit_done=is_explicit_done,
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
        analysis, _inhouse_signal = await _analyze_narrative(narrative_text, turns=[u.text for u in state.utterances if u.role == "user"])
        state.narrative_analysis = analysis
        state.phase = ConversationPhase.structured

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
            # Free-text clinical slot extraction
            if user_text:
                extracted_slots = NLPEngine.extract_clinical_slots(user_text)

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
