# Project structure

```text
mentality/
|-- app.py                         Application entrypoint
|-- index.html                     Browser interface
|-- mindtriage/
|   |-- api.py                     FastAPI routes and model orchestration
|   |-- config.py                  Environment configuration
|   |-- dataset.py                 Consented JSONL logger and redaction
|   |-- dialogue.py                Slot state and MI policy selection
|   |-- generation.py              Groq and local LoRA generation
|   |-- nlp.py                     Neural transformer inference
|   |-- persistence.py             SQLite/PostgreSQL anonymized analytics
|   |-- reports.py                 Non-diagnostic summaries
|   |-- risk.py                    Neural risk inference and risk ordering
|   |-- safety.py                  Independent emergency fail-safe
|   |-- schemas.py                 API contracts
|   `-- sessions.py                Redis/memory session stores
|-- train_classifier.py            MentalBERT risk training and evaluation
|-- train_nlp.py                   Transformer emotion/topic/slot training
|-- train_dialogue.py              TinyLlama LoRA fine-tuning
|-- benchmark_risk_models.py       MentalBERT/DistilBERT comparison
|-- train_all_models.py            Reproducible complete training pipeline
|-- MODEL_TRAINING.md              Exact model-training runbook
|-- scripts/
|   |-- download_mentalbert.py     Gated base-model download
|   `-- generate_dialogue_scenarios.py  Groq teacher-data generation
|-- tests/                         Unit and integrated API tests
|-- data/                          Runtime database and consented JSONL data
|-- models/                        Base and trained model artifacts
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- .env.example
```

`outputs/` and the original workbook are research/training artifacts, not active-session storage.
