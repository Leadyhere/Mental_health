# MindTriage

MindTriage is a privacy-conscious, non-diagnostic conversational support and risk-triage system. Normal language understanding, risk classification, and response generation are model-based. A deterministic emergency guard exists only as a final safety override.

> This software is not a therapist, diagnostic device, or emergency service. Safety and clinical deployment require independent professional validation, legal review, monitored operations, and verified regional crisis resources.

## Implemented

- Responsive browser chat, consent control, report download, and explicit session deletion
- FastAPI API with validated request/response schemas
- Redis sessions with TTL and safe in-memory development fallback
- SQLite development analytics and PostgreSQL production support
- No raw chat text in analytics tables; session IDs are HMAC-hashed
- Opt-in JSONL dialogue dataset with email, phone, and URL redaction
- Three trainable transformer NLP heads for emotion, topic, and BIO semantic-slot extraction
- Weighted slot selection, two-turn cooldown, and MI/OARS dialogue policies
- MentalBERT five-class inference; missing weights fail closed with HTTP 503
- Cross-turn highest-risk retention
- Deterministic active/passive self-harm rules and controlled emergency response
- Groq `openai/gpt-oss-120b` or a local TinyLlama LoRA dialogue model
- Full MentalBERT training/evaluation pipeline using all Q1-Q15 fields
- LoRA dialogue fine-tuning pipeline using consented JSONL records
- Docker, Redis, PostgreSQL, health checks, and automated tests

## Local setup

Use Python 3.12 or newer. Dependencies are pinned to the versions used by the
functional test suite.

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Open `http://localhost:8000`. The API reports `degraded` and returns HTTP 503 for chat until the required NLP/risk weights and either Groq or local dialogue inference are available.

## Production services

Update `.env` with a strong `ANALYTICS_HASH_SECRET`, Groq key, and production database URL. Then run:

```powershell
docker compose up --build
```

Production mode requires PostgreSQL. Redis stores active conversations with a configurable TTL; PostgreSQL stores only anonymized operational records.

## Train the risk classifier

The workbook contains 11,500 labelled rows across Q1-Q15. Training removes exact duplicates, then uses stratified 70/15/15 splits, balanced class weights, AdamW, early stopping, per-class metrics, confusion matrix, and emergency recall.

```powershell
hf auth login
python scripts/download_mentalbert.py
python train_classifier.py --base-model models/base/mental-bert-base-uncased
```

Outputs are written to `models/inhouse_risk_classifier/` and loaded automatically. The optional 100,000-row synthetic CSV contributes only its 80,000 preassigned training rows. Source-group IDs align its paraphrases with the real workbook split, so validation and test remain real and no source group crosses partitions:

```powershell
python train_classifier.py --base-model models/base/mental-bert-base-uncased --synthetic-dataset outputs/dataset-v2/mental_health_synthetic_v2_100k_report.csv
python benchmark_risk_models.py
```

The benchmark trains MentalBERT and DistilBERT independently on identical splits. It does not silently substitute one model for another.

## Train the deep-learning NLP extractors

Emotion, problem-topic, and semantic-slot detection use three fine-tuned transformer heads built from the Q1-Q15 workbook:

```powershell
python train_nlp.py --base-model models/base/mental-bert-base-uncased
```

Outputs are written to `models/inhouse_nlp/{emotion,topic,slots}` and loaded automatically. There is no keyword-based NLP fallback.

## Train the local dialogue model

Only conversations where the user enabled training consent are written to `data/conversations.jsonl`.

```powershell
python train_dialogue.py
```

The trainer excludes deterministic emergency messages, removes duplicate pairs, separates validation by conversation ID, masks prompt tokens, and produces a LoRA adapter in `models/inhouse_llama_dialogue/`.

For project-specific dialogue data, replay scenario cards through the trained NLP/risk pipeline and Groq:

```powershell
python scripts/generate_dialogue_scenarios.py --conversations 1000
python train_dialogue.py --dataset data/groq_dialogues.jsonl
```

This writes `data/groq_dialogues.jsonl`. Generation stops if a required neural model or Groq is unavailable. Review it before training, and keep a separate safety-test set that is never used as training data.

## Test

```powershell
python -m unittest discover -s tests -v
```

All model trainers show batch-level percentage and ETA in the terminal and log metrics to
TensorBoard under `runs/`. Launch the dashboard with `tensorboard --logdir runs --port 6006`.

## Key API routes

| Route | Purpose |
|---|---|
| `POST /chat/start` | Create a temporary session and record consent choice |
| `POST /chat/message` | Process one conversation turn |
| `POST /chat/summary` | Build a non-diagnostic summary without putting the session ID in a URL |
| `POST /chat/end` | Delete active session data |
| `GET /health` | Report runtime component status |

See [MODEL_TRAINING.md](MODEL_TRAINING.md), [ARCHITECTURE.md](ARCHITECTURE.md), [TECH_STACK.md](TECH_STACK.md), and [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details.
