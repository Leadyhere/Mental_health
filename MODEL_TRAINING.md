# Model training runbook

## Live progress and TensorBoard

Every trainer now displays a batch progress bar with:

- percentage for the current epoch;
- overall percentage across all planned epochs;
- current loss, elapsed time, processing rate, and ETA.

Progress refreshes at most once per second and after a batch completes. A very slow batch can
therefore take longer than one second before the next visible update.

TensorBoard logging is enabled by default. Start the dashboard in a second PowerShell terminal:

```powershell
tensorboard --logdir runs --port 6006
```

Then open `http://localhost:6006`. The dashboard includes batch loss, epoch training and
validation metrics, completion percentage, and test metrics. Each run is stored in a new
timestamped folder, so earlier training runs are preserved.

Optional controls:

```powershell
# Choose a specific dashboard folder
python train_nlp.py --tensorboard-dir runs/my_nlp_run

# Log TensorBoard scalars every 10 batches instead of every batch
python train_nlp.py --log-every-steps 10

# Disable TensorBoard while retaining terminal percentage/ETA bars
python train_nlp.py --no-tensorboard
```

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

Training now removes exact duplicate texts before splitting, so identical examples cannot
appear in both training and evaluation. These metrics still measure questionnaire-derived
language only; use a separately annotated free-form chat holdout before claiming real-world
NLP performance. Existing NLP weights created by an older script should be retrained.
For topic training, labels with fewer than 10 unique examples remain in training only and
are recorded under `train_only_rare_labels` in `metrics.json`; they are not included in
holdout claims until more independently labelled examples are collected.

## 3. MentalBERT risk model

This trains the five-level risk classifier using weighted cross-entropy, AdamW, warmup, early stopping, and a stratified 70/15/15 split:

```powershell
python train_classifier.py --base-model models/base/mental-bert-base-uncased
```

To use the project-specific synthetic file, the trainer aligns its source-group split with
the real workbook and adds only the 80,000 rows marked `train`:

```powershell
python train_classifier.py --base-model models/base/mental-bert-base-uncased --synthetic-dataset outputs/dataset-v2/mental_health_synthetic_v2_100k_report.csv
```

The real workbook remains the validation and test source. Review macro F1, per-class recall, the confusion matrix, and especially emergency recall in `models/inhouse_risk_classifier/metrics.json`.

Without synthetic augmentation, exact duplicate questionnaire rows are removed before the
70/15/15 split. With the 100k file, its source-group metadata defines an aligned 80/10/10
partition and only synthetic `train` rows are used. The saved metrics do not replace a
conversational or clinical holdout evaluation. Existing risk weights created before this
deduplication change should be retrained.

## 4. DistilBERT comparison

```powershell
python benchmark_risk_models.py
```

This trains MentalBERT and DistilBERT separately on identical data splits and writes `models/risk_benchmarks/comparison.json`. It is a benchmark, not a hidden runtime fallback.

## 5. Groq teacher data and the in-house dialogue model

Add `GROQ_API_KEY` to `.env`, train NLP and risk first, then generate project-specific conversations:

```powershell
python scripts/generate_dialogue_scenarios.py --conversations 1000 --compact --model llama-3.1-8b-instant
python train_dialogue.py --dataset data/groq_dialogues.jsonl
```

The local model is TinyLlama 1.1B fine-tuned with supervised LoRA. Enable it after review with `ENABLE_LOCAL_DIALOGUE_MODEL=true`; it then takes priority over Groq.

## 6. Verification

```powershell
python -m unittest discover -s tests -v
python app.py
```

`GET /health` must show the NLP models and risk model as ready with
`deduplicated-questionnaire-holdout` validation. Older weights are marked
`legacy-metrics-retrain-required`. No production classifier falls back to keywords or
templates when weights are missing.

For a scheduled full run after Groq data exists:

```powershell
python train_all_models.py --with-synthetic-risk --with-benchmark --with-dialogue
```
