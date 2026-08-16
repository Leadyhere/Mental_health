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
