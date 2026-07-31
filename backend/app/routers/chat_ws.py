from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.conversation.state_machine import handle_client_message
from app.deps import verify_session_token
from app.llm.safety_filter import filter_outbound_message
from app.models.schemas import ClientMessage
from app.services.session_store import get_session_store

router = APIRouter(tags=["chat"])


@router.websocket("/api/sessions/{session_id}/chat")
async def chat_ws(websocket: WebSocket, session_id: str, token: str):
    if verify_session_token(token) != session_id:
        await websocket.close(code=4401)
        return

    store = get_session_store()
    state = await store.get(session_id)
    if state is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()

    # Replay the orientation message so a fresh client connection sees it.
    for utterance in state.utterances:
        if utterance.role == "assistant":
            await websocket.send_json({"type": "assistant_text", "payload": {"text": utterance.text}})

    try:
        while True:
            raw = await websocket.receive_json()
            message = ClientMessage.model_validate(raw)

            server_messages = await handle_client_message(state, message)
            await store.save(state)

            for server_message in server_messages:
                if server_message.type == "assistant_text":
                    server_message.payload["text"] = await filter_outbound_message(server_message.payload["text"])
                await websocket.send_json(server_message.model_dump(mode="json"))
    except WebSocketDisconnect:
        await store.save(state)
