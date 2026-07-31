from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.conversation.narrative_phase import orientation_message
from app.deps import issue_session_token
from app.models.schemas import ConversationPhase, ServerMessage, SessionState, Utterance
from app.services.report_service import build_report, build_report_pdf
from app.services.session_store import get_session_store
from app.services.teardown_service import teardown_session

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("")
async def create_session():
    state = SessionState(phase=ConversationPhase.narrative)
    orientation_text = orientation_message()
    state.utterances.append(Utterance(turn_id=1, role="assistant", text=orientation_text))

    await get_session_store().save(state)
    token = issue_session_token(str(state.session_id))

    return {
        "session_id": str(state.session_id),
        "token": token,
        "orientation_message": orientation_text,
    }


@router.get("/{session_id}/report")
async def get_report(session_id: str, format: str | None = None):
    state = await get_session_store().get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found or already ended")
    if state.current_risk_assessment is None:
        raise HTTPException(status_code=409, detail="session has not reached a risk assessment yet")

    report = build_report(state)
    if format == "pdf":
        pdf_bytes = build_report_pdf(report)
        return Response(content=pdf_bytes, media_type="application/pdf")
    return report


@router.post("/{session_id}/end")
async def end_session(session_id: str):
    state = await get_session_store().get(session_id)
    if state is None:
        return {"status": "already ended"}
    await teardown_session(state)
    return {"status": "ended"}
