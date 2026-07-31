"""Stage 2 periodic retrain job, triggered every N new sessions (see
backend/app/routers/admin.py POST /api/admin/retrain, and
RetrainCounter in backend/app/models/db.py).

Current scope note: live-session structured answers use question ids
(emotional_clarification, coping_behaviors, ...) rather than the original
dataset's Q1-Q15 columns, so they can't yet be concatenated 1:1 into
train.parquet's schema for supervised fine-tuning -- that mapping is a
follow-up task. For now this job retrains on the original bootstrap
dataset (re-affirming the checkpoint/versioning pipeline works end to end)
and then runs the agreement-rate eval against every logged LLM decision
using the model's generic `predict(turns: list[str])` path, which works
regardless of schema. Once the schema mapping exists, swap the `train_df`
below for a concatenation of train.parquet + a schema-mapped version of
data/processed/llm_decisions.jsonl.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from inhouse_model import config
from inhouse_model.model import InHouseModel
from scripts.eval_agreement import run_agreement_eval


def run_retrain() -> dict:
    train_df = pd.read_parquet(config.TRAIN_PARQUET)
    eval_df = pd.read_parquet(config.EVAL_PARQUET)

    model = InHouseModel()
    metadata = model.train(train_df, eval_dataset=eval_df)

    agreement = run_agreement_eval()

    _write_model_version_row(metadata["version"], agreement)

    return {"version": metadata["version"], "agreement": agreement}


def _write_model_version_row(version: int, agreement: dict) -> None:
    import asyncio

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.models.db import ModelVersion, get_session_maker

    async def _write():
        session_maker = get_session_maker()
        async with session_maker() as db_session:
            db_session.add(
                ModelVersion(
                    version=version,
                    agreement_rate=agreement.get("agreement_rate"),
                    eval_sample_size=agreement.get("n"),
                )
            )
            await db_session.commit()

    asyncio.run(_write())


if __name__ == "__main__":
    run_retrain()
