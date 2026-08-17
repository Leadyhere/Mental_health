from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .config import Settings, get_settings
from .dataset import ConsentedDatasetLogger
from .dialogue import DialogueStateTracker
from .generation import ConversationGenerator, ConversationModelUnavailable
from .nlp import NLPModelUnavailable, build_feature_extractor
from .persistence import AnalyticsRepository
from .reports import ReportBuilder
from .risk import RiskClassifier, RiskModelUnavailable, highest_risk
from .safety import SafetyEngine
from .schemas import (
    ChatResponse,
    EndSessionRequest,
    MessageRequest,
    StartSessionRequest,
    SummaryRequest,
    SummaryResponse,
)
from .sessions import build_session_store


def create_app(
    settings: Settings | None = None,
    feature_extractor=None,
    risk_model=None,
    conversation_model=None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.validate()
    app = FastAPI(
        title="MindTriage API",
        version="2.1.0",
        description="Privacy-conscious, non-diagnostic conversational support and triage API",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    sessions = build_session_store(settings)
    analytics = AnalyticsRepository(settings)
    dataset_logger = ConsentedDatasetLogger(settings.dataset_path)
    extractor = feature_extractor or build_feature_extractor(settings)
    tracker = DialogueStateTracker()
    risk_classifier = risk_model or RiskClassifier(settings)
    safety = SafetyEngine()
    generator = conversation_model or ConversationGenerator(settings)
    reports = ReportBuilder()

    app.state.settings = settings
    app.state.sessions = sessions
    app.state.analytics = analytics
    app.state.risk_classifier = risk_classifier
    app.state.generator = generator

    @app.post("/chat/start")
    def start_chat(req: StartSessionRequest | None = None):
        request = req or StartSessionRequest()
        session = sessions.create(request.training_consent, request.locale)
        analytics.session_started(session)
        return {
            "session_id": session["session_id"],
            "status": "active",
            "training_consent": session["training_consent"],
            "expires_in_seconds": settings.session_ttl_seconds,
        }

    @app.post("/chat/message", response_model=ChatResponse)
    def chat_message(req: MessageRequest):
        session = sessions.get(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        user_text = req.user_input.strip()
        if not user_text:
            raise HTTPException(status_code=422, detail="Message cannot be empty")
        if len(user_text) > settings.max_message_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Message exceeds the {settings.max_message_chars}-character limit",
            )

        try:
            features = extractor.extract(user_text)
        except NLPModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        session["filled_slots"] = tracker.update(
            session["filled_slots"], features, user_text
        )
        dialogue_decision = tracker.decide(
            session["filled_slots"], session["turn_history"]
        )
        try:
            prediction = risk_classifier.predict(
                user_text, features, session["chat_history"]
            )
        except RiskModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        safety_decision = safety.evaluate(user_text, prediction.level, features)

        if safety_decision.is_emergency:
            bot_reply = safety.emergency_response()
            model_source = "deterministic-safety-v2"
        else:
            try:
                bot_reply = generator.generate(
                    user_text,
                    features,
                    dialogue_decision,
                    session["chat_history"],
                    safety_decision.final_risk,
                    safety_decision.reason_codes,
                )
            except ConversationModelUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            model_source = generator.source

        session["turn_count"] += 1
        session["turn_history"].append(dialogue_decision.target_slot)
        session["current_risk_level"] = safety_decision.final_risk
        session["highest_risk_level"] = highest_risk(
            session["highest_risk_level"], safety_decision.final_risk
        )
        session["chat_history"].append({"user": user_text, "bot": bot_reply})
        if not sessions.save(session):
            raise HTTPException(
                status_code=409,
                detail="Session ended or expired while the message was processing",
            )

        analytics.risk_event(
            session["session_id"],
            safety_decision.final_risk,
            safety_decision.is_emergency,
            prediction.source,
            safety_decision.reason_codes,
        )
        if session["training_consent"]:
            dataset_logger.log(
                session["dataset_conversation_id"],
                session["turn_count"],
                user_text,
                bot_reply,
                features.as_dict(),
                safety_decision.final_risk,
                dialogue_decision.target_slot,
                dialogue_decision.policy,
                model_source,
            )

        return ChatResponse(
            session_id=session["session_id"],
            bot_reply=bot_reply,
            risk_level=safety_decision.final_risk,
            highest_risk_level=session["highest_risk_level"],
            is_emergency=safety_decision.is_emergency,
            emergency_contacts=safety_decision.contacts,
            extracted_emotions=features.emotions,
            target_slot=dialogue_decision.target_slot,
            model_source=model_source,
        )

    @app.post("/chat/summary", response_model=SummaryResponse)
    def summary(req: SummaryRequest):
        session = sessions.get(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        report, concerns = reports.build(session)
        return SummaryResponse(
            session_id=req.session_id,
            report=report,
            risk_level=session["highest_risk_level"],
            concerns=concerns,
        )

    @app.post("/chat/end")
    def end_chat(req: EndSessionRequest):
        deleted = sessions.delete(req.session_id)
        if deleted:
            analytics.session_ended(req.session_id)
        return {
            "session_id": req.session_id,
            "status": "deleted" if deleted else "already_deleted",
        }

    @app.get("/health")
    def health():
        model_stack_ready = all(
            getattr(component, "ready", True)
            for component in (extractor, risk_classifier, generator)
        ) and all(
            getattr(component, "validated", True)
            for component in (extractor, risk_classifier)
        )
        return {
            "status": "ok" if analytics.health() and model_stack_ready else "degraded",
            "session_store": sessions.name,
            "database": analytics.backend,
            "nlp_models": "ready" if getattr(extractor, "ready", True) else "unavailable",
            "nlp_validation": getattr(extractor, "validation_status", "test-or-external"),
            "conversation_model": generator.source,
            "risk_model": risk_classifier.model_version,
            "risk_validation": getattr(risk_classifier, "validation_status", "test-or-external"),
        }

    @app.get("/", response_class=HTMLResponse)
    def frontend():
        if not settings.frontend_path.exists():
            raise HTTPException(status_code=404, detail="Frontend not found")
        return settings.frontend_path.read_text(encoding="utf-8")

    return app
