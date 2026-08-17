# Technology stack

| Layer | Technology | Function |
|---|---|---|
| Frontend | HTML, CSS, JavaScript | Accessible chat, consent, safety banner, and report download |
| API | Python, FastAPI, Pydantic | Validation and orchestration |
| Session data | Redis | Temporary active conversation storage with TTL |
| Analytics | PostgreSQL; SQLite locally | Anonymized sessions and risk events |
| Conversation model | Groq GPT-OSS 120B (low reasoning effort) or TinyLlama LoRA | MI/OARS-style conversational wording |
| Local dialogue | TinyLlama 1.1B + PEFT LoRA | In-house generative response model |
| Risk model | Fine-tuned MentalBERT | Five-level risk classification |
| NLP extraction | Three fine-tuned transformer heads | Emotion, problem-topic, and semantic-slot detection |
| Model comparison | DistilBERT benchmark | Independent baseline against MentalBERT |
| Safety | Deterministic emergency guard | Fail-safe override only; not a normal classifier |
| Training | PyTorch, Transformers, scikit-learn | Classifier training and evaluation |
| Dialogue training | PEFT LoRA | Efficient local causal-LM adaptation |
| Training monitoring | tqdm, TensorBoard | Batch percentage, ETA, loss curves, and metrics dashboard |
| Packaging | Docker Compose | API, PostgreSQL, and Redis services |
| Testing | unittest, FastAPI TestClient | Component and integrated API verification |

The MentalBERT model is gated and uses a non-commercial licence. Confirm access and licence compatibility before distribution or commercial deployment.
