# Local Safety Testing Report

## 2026-08-13 01:02 IST

- Hosted-model access disabled (`GROQ_API_KEY` unset); no conversation data uploaded and no model weights changed.
- Python compilation passed and all 14 unit/API tests passed.
- Replayed 20 local synthetic conversations: 4 each for Low, Mild, Moderate, Severe, and Emergency.
- Emergency detection, explicit-negation handling, cross-turn highest-risk retention, consent/redaction, session deletion, summaries, and analytics privacy all passed.

Result: **PASS**

## 2026-08-14 01:16 IST

- Hosted-model access disabled (`GROQ_API_KEY` unset); no conversation data uploaded and no model weights changed.
- Python compilation passed and all 14 unit/API tests passed.
- Replayed 20 local synthetic conversations: 4 each for Low, Mild, Moderate, Severe, and Emergency.
- Emergency detection, explicit-negation handling, cross-turn highest-risk retention, consent/redaction, session deletion, summaries, and analytics privacy all passed.

Result: **PASS**

## 2026-08-16 11:26 IST

- Hosted-model access disabled (`GROQ_API_KEY` unset); no conversation data uploaded and no model weights changed.
- Python compilation passed and all 14 unit/API tests passed.
- Replayed 20 local synthetic conversations: 4 each for Low, Mild, Moderate, Severe, and Emergency; every replay retained its highest risk across a follow-up turn.
- Emergency detection, explicit-negation handling, consent/redaction, session deletion, summaries, and analytics privacy all passed. No consented dataset was created during replay.

Result: **PASS**

## 2026-08-16 deep functional audit

- Expanded automated coverage from 14 to 28 unit/integrated API tests; all pass.
- Python compilation, installed dependency integrity, pinned-version matching, Docker Compose
  configuration, HTML/JavaScript parsing, and tracked-file secret/artifact scans pass.
- Real MentalBERT and all three NLP models load and perform inference.
- A synthetic end-to-end request passed through the real NLP stack, risk model, session store,
  FastAPI routes, Groq dialogue model, summary generation, and session deletion.
- Safety regressions fixed: model Level 5 always uses the controlled emergency response, mixed
  negation cannot mask a later active threat, and tested quoted/negated phrases avoid false alarms.
- Privacy/reliability regressions fixed: nested extracted features are redacted, summary session IDs
  no longer appear in URLs, failed requests do not partially mutate memory sessions, and late saves
  cannot recreate a deleted session.
- Training leakage fixed: 3,230 duplicate real risk texts and 45,562 duplicate slot samples are now
  removed before splitting; synthetic augmentation is aligned by all 8,270 source-group IDs.
- Existing saved NLP/risk weights are intentionally reported as
  `legacy-metrics-retrain-required`; retraining is required before health returns `ok`.
- Risk, NLP, and dialogue trainers now provide batch-level epoch/overall percentages,
  current loss, overall ETA, and TensorBoard event logs; event-file creation is regression-tested.

Result: **CODE AND RUNTIME PASS; MODEL RETRAIN REQUIRED FOR VALIDATED METRICS**

## 2026-08-17 01:02 IST

- Hosted-model access disabled (`GROQ_API_KEY` unset); no conversation data uploaded and no model weights changed.
- Python compilation passed and all 28 unit/API tests passed.
- Replayed 20 fresh local synthetic API conversations: 4 each for Low, Mild, Moderate, Severe, and Emergency. Each retained its highest risk across a lower-risk follow-up.
- Emergency routing, explicit negation, summaries, idempotent session deletion, consent/redaction, and analytics privacy checks passed. Non-consented replay data was not written to the local dataset.

Result: **PASS**
