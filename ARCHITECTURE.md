# Architecture

MindTriage is a modular monolith. FastAPI contains separately testable modules; Redis, PostgreSQL, Groq, and local model inference remain replaceable adapters.

## Request flow

```text
Browser
  -> FastAPI validation
  -> Redis session lookup
  -> transformer emotion/topic/BIO slot inference
  -> dialogue-state decision
  -> MentalBERT five-level risk inference
  -> independent emergency safety guard
       emergency -> fixed verified crisis response
       otherwise -> Groq Llama or local LoRA LLM response
  -> update cross-turn state
  -> anonymized analytics
  -> consented/redacted JSONL record (optional)
  -> browser response
```

## Components

1. **Session store:** Redis `SETEX` records expire automatically. Memory storage is local-development only.
2. **NLP inference:** three fine-tuned transformer heads predict emotions, topic, and token-level semantic slots.
3. **Dialogue tracker:** fills trigger, duration, impact, coping, support, and emotion state.
4. **Risk classifier:** fine-tuned MentalBERT returns one of five ordered categories with probabilities and a model version.
5. **Safety engine:** an independent deterministic guard can only escalate emergencies and block generative output; it is not the normal classifier.
6. **Conversation generator:** Groq Llama 3.3 or a locally fine-tuned TinyLlama LoRA adapter uses bounded MI/OARS prompts.
7. **Analytics repository:** stores HMAC session hashes, consent, risk events, model versions, and reason codes, never raw messages.
8. **Dataset logger:** writes only explicitly consented, redacted JSONL turns.

## Data boundaries

| Data | Location | Retention |
|---|---|---|
| Raw active conversation | Redis | TTL or explicit end |
| Anonymized operational events | PostgreSQL | Organizational policy |
| Consented redacted training turns | JSONL/object storage | Dataset governance policy |
| Model weights and metrics | Model registry/filesystem | Versioned |

## Safety properties

- Emergency responses are deterministic and never composed by an LLM.
- Highest observed risk is retained across the session.
- Analytics contain no raw conversation text.
- Training consent is optional and disabled by default.
- Missing NLP/risk/dialogue models never trigger a heuristic or template fallback.

Implementation completeness does not equal clinical readiness. Real-user deployment still requires expert review, crisis-resource verification, adversarial safety evaluation, privacy/security assessment, bias analysis, and monitored rollout.
