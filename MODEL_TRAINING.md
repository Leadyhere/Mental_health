# Model training runbook

## 1. Base model assets

MentalBERT is already stored at `models/base/mental-bert-base-uncased`. To restore it later:

```powershell
hf auth login
python scripts/download_mentalbert.py
```

## 2. Neural NLP models

This trains three independent MentalBERT heads from the 11,500-row workbook:

- multi-label emotion classification with sigmoid/BCE loss;
- eight-class problem-topic classification with softmax/cross-entropy;
- BIO token classification for trigger, duration, impact, coping, and support.

```powershell
python train_nlp.py --base-model models/base/mental-bert-base-uncased
```

Review each `metrics.json` under `models/inhouse_nlp/`.

## 3. MentalBERT risk model

This trains the five-level risk classifier using weighted cross-entropy, AdamW, warmup, early stopping, and a stratified 70/15/15 split:

```powershell
python train_classifier.py --base-model models/base/mental-bert-base-uncased
```

To add the 100,000 project-specific synthetic rows to training only:

```powershell
python train_classifier.py --base-model models/base/mental-bert-base-uncased --synthetic-dataset outputs/dataset-v2/mental_health_synthetic_v2_100k_report.csv
```

The real workbook remains the validation and test source. Review macro F1, per-class recall, the confusion matrix, and especially emergency recall in `models/inhouse_risk_classifier/metrics.json`.

## 4. DistilBERT comparison

```powershell
python benchmark_risk_models.py
```

This trains MentalBERT and DistilBERT separately on identical data splits and writes `models/risk_benchmarks/comparison.json`. It is a benchmark, not a hidden runtime fallback.

## 5. Groq teacher data and the in-house dialogue model

Add `GROQ_API_KEY` to `.env`, train NLP and risk first, then generate project-specific conversations:

```powershell
python scripts/generate_dialogue_scenarios.py --conversations 1000
python train_dialogue.py --dataset data/groq_dialogues.jsonl
```

The local model is TinyLlama 1.1B fine-tuned with supervised LoRA. Enable it after review with `ENABLE_LOCAL_DIALOGUE_MODEL=true`; it then takes priority over Groq.

## 6. Verification

```powershell
python -m unittest discover -s tests -v
python app.py
```

`GET /health` must show the NLP models and risk model as ready. No production classifier falls back to keywords or templates when weights are missing.

For a scheduled full run after Groq data exists:

```powershell
python train_all_models.py --with-synthetic-risk --with-benchmark --with-dialogue
```
