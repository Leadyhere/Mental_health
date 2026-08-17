"""Fine-tune the local dialogue model with LoRA on consented JSONL turns."""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindtriage.training import TrainingMonitor, timestamped_run_dir


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "conversations.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "models" / "inhouse_llama_dialogue"
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dialogue dataset not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))

    cleaned = []
    seen = set()
    for index, record in enumerate(records):
        user_text = record.get("user_text") or record.get("user_natural_input")
        assistant_text = record.get("assistant_text") or record.get("groq_generated_reply")
        model_source = record.get("conversation_model", "legacy")
        excluded_source = model_source.startswith("deterministic-safety") or "controlled-template" in model_source
        if not user_text or not assistant_text or excluded_source:
            continue
        key = (user_text.strip(), assistant_text.strip())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "conversation_id": record.get("conversation_id", f"legacy_{index}"),
                "user_text": user_text.strip(),
                "assistant_text": assistant_text.strip(),
                "risk_level": record.get("risk_level")
                or record.get("observed_risk")
                or "unknown",
            }
        )
    return cleaned


def grouped_split(records: list[dict], validation_ratio: float, seed: int):
    conversation_ids = sorted({record["conversation_id"] for record in records})
    if len(conversation_ids) < 2:
        raise ValueError(
            "Dialogue training requires at least two distinct conversation IDs so validation "
            "does not reuse the training conversation."
        )
    random.Random(seed).shuffle(conversation_ids)
    validation_count = min(
        len(conversation_ids) - 1,
        max(1, int(len(conversation_ids) * validation_ratio)),
    )
    validation_ids = set(conversation_ids[:validation_count])
    train_records = [record for record in records if record["conversation_id"] not in validation_ids]
    validation_records = [record for record in records if record["conversation_id"] in validation_ids]
    return train_records, validation_records


class DialogueDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        prompt = (
            "<|system|>\nProvide non-diagnostic emotional support. "
            f"Internal safety category: {record['risk_level']}. "
            "For severe concern, check safety and encourage prompt human support. "
            "Do not reveal internal labels.</s>\n"
            f"<|user|>\n{record['user_text']}</s>\n<|assistant|>\n"
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        response_ids = self.tokenizer(
            record["assistant_text"] + "</s>", add_special_tokens=False
        )["input_ids"]
        if not response_ids:
            raise ValueError("Dialogue record produced no assistant tokens")
        response_ids = response_ids[: self.max_length]
        prompt_budget = self.max_length - len(response_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
        unpadded = prompt_ids + response_ids
        padding = self.max_length - len(unpadded)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.tensor(unpadded + [pad_id] * padding, dtype=torch.long)
        attention_mask = torch.tensor(
            [1] * len(unpadded) + [0] * padding, dtype=torch.long
        )
        labels = torch.tensor(
            [-100] * len(prompt_ids) + response_ids + [-100] * padding,
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def validation_loss(
    model, loader, device, monitor: TrainingMonitor | None = None,
    epoch: int | None = None,
):
    model.eval()
    total = 0.0
    batches = monitor.evaluation_batches(loader, "validation", epoch) if monitor else loader
    with torch.no_grad():
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            total += model(**batch).loss.item()
    return total / max(len(loader), 1)


def train(args):
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch size must be greater than zero")
    if args.learning_rate <= 0 or args.max_length <= 0:
        raise ValueError("learning rate and max length must be greater than zero")
    if args.log_every_steps <= 0:
        raise ValueError("log-every-steps must be greater than zero")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.dataset)
    if len(records) < args.minimum_records and not args.allow_small_dataset:
        raise ValueError(
            f"Only {len(records)} valid turns found; at least {args.minimum_records} are required. "
            "Use --allow-small-dataset only for pipeline testing."
        )
    train_records, validation_records = grouped_split(records, 0.1, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_loader = DataLoader(
        DialogueDataset(train_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        DialogueDataset(validation_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    patience_remaining = args.patience
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = args.tensorboard_dir or timestamped_run_dir(BASE_DIR, "dialogue")
    monitor = TrainingMonitor(
        run_dir=run_dir,
        task_name="dialogue-lora",
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        log_every_steps=args.log_every_steps,
        tensorboard_enabled=not args.no_tensorboard,
    )
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        progress = monitor.training_batches(train_loader, epoch + 1)
        for batch_number, batch in enumerate(progress, start=1):
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            monitor.log_batch(progress, epoch + 1, batch_number, float(loss.item()))
        result = {
            "epoch": epoch + 1,
            "train_loss": total / max(len(train_loader), 1),
            "validation_loss": validation_loss(
                model, validation_loader, device, monitor, epoch + 1
            ),
        }
        history.append(result)
        monitor.log_epoch(
            epoch + 1,
            {
                "train_loss": result["train_loss"],
                "validation_loss": result["validation_loss"],
            },
        )
        print(json.dumps(result), flush=True)
        if result["validation_loss"] < best_validation_loss:
            best_validation_loss = result["validation_loss"]
            best_epoch = epoch + 1
            patience_remaining = args.patience
            model.save_pretrained(args.output)
        else:
            patience_remaining -= 1
            if patience_remaining == 0:
                break

    monitor.finish(len(history))
    monitor.close()
    tokenizer.save_pretrained(args.output)
    metrics = {
        "base_model": args.base_model,
        "algorithm": "LoRA supervised causal language-model fine-tuning",
        "records": len(records),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_conversations": len({record["conversation_id"] for record in train_records}),
        "validation_conversations": len({record["conversation_id"] for record in validation_records}),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "tensorboard_run": str(run_dir) if not args.no_tensorboard else None,
        "history": history,
    }
    (args.output / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--minimum-records", type=int, default=100)
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
