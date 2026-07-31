"""Stage 2 core metric: how often the in-house model's risk_level prediction
matches the LLM's, on the logged LLM-decision triples (data/processed/
llm_decisions.jsonl, written by InHouseModel.log_llm_decision during live
sessions -- see backend/app/services/logging_service.py).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inhouse_model import config, metrics
from inhouse_model.model import InHouseModel


def load_llm_decisions() -> list[dict]:
    if not config.LLM_DECISIONS_PATH.exists():
        return []
    with open(config.LLM_DECISIONS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_agreement_eval() -> dict:
    decisions = load_llm_decisions()
    if not decisions:
        print("No logged LLM decisions yet -- nothing to evaluate.")
        return {"agreement_rate": None, "n": 0}

    model = InHouseModel().load_latest()
    inhouse_preds = []
    llm_preds = []
    for record in decisions:
        risk_level, _sub_labels, _confidence = model.predict(record["turns"])
        inhouse_preds.append(risk_level)
        llm_preds.append(record["llm_risk_level"])

    rate = metrics.agreement_rate(inhouse_preds, llm_preds)
    print(f"Agreement rate over {len(decisions)} logged sessions: {rate:.4f}")
    return {"agreement_rate": rate, "n": len(decisions)}


if __name__ == "__main__":
    run_agreement_eval()
