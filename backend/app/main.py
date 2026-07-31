from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.crisis_resources import CRISIS_RESOURCES
from app.models.db import init_db
from app.routers import admin, auth, chat_ws, sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Mental Health Conversational Triage Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(chat_ws.router)
app.include_router(admin.router)
app.include_router(auth.router)


@app.get("/api/crisis-resources")
async def crisis_resources():
    return CRISIS_RESOURCES


@app.get("/health")
async def health():
    return {"status": "ok"}


# Ephemeral session backstop: session_store.py sets a Redis TTL equal to
# SESSION_IDLE_TIMEOUT_MINUTES on every save, so an abandoned session's raw
# conversation is dropped by Redis itself even if the client never calls
# POST /api/sessions/{id}/end -- no separate polling task needed.
