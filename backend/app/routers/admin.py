from __future__ import annotations

import asyncio

from fastapi import APIRouter
from sqlalchemy import select

from app.models.db import ModelVersion, RetrainCounter, get_session_maker

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/retrain")
async def trigger_retrain():
    """Manually invoke the Stage 2 retrain job. No auth UI yet for the
    human-review pass (deferred per plan) -- this is a minimal, unauthenticated
    trigger suitable only for local development.

    Runs in a worker thread: retrain_stage2.run_retrain() is a synchronous,
    CPU-bound training routine that internally does its own asyncio.run()
    for the ModelVersion write, which cannot happen on this request's
    already-running event loop.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.retrain_stage2 import run_retrain

    result = await asyncio.to_thread(run_retrain)

    session_maker = get_session_maker()
    async with session_maker() as db_session:
        counter = await db_session.get(RetrainCounter, 1)
        if counter is not None:
            counter.sessions_since_last_retrain = 0
        await db_session.commit()

    return result


@router.get("/agreement-metrics")
async def agreement_metrics():
    session_maker = get_session_maker()
    async with session_maker() as db_session:
        result = await db_session.execute(select(ModelVersion).order_by(ModelVersion.version.desc()))
        versions = result.scalars().all()
    return [
        {
            "version": v.version,
            "agreement_rate": v.agreement_rate,
            "eval_sample_size": v.eval_sample_size,
            "created_at": v.created_at.isoformat(),
        }
        for v in versions
    ]
