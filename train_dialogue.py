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
            }
        )
    return cleaned


def grouped_split(records: list[dict], validation_ratio: float, seed: int):
    conversation_ids = sorted({record["conversation_id"] for record in records})
    random.Random(seed).shuffle(conversation_ids)
    validation_count = max(1, int(len(conversation_ids) * validation_ratio))
    validation_ids = set(conversation_ids[:validation_count])
    train_records = [record for record in records if record["conversation_id"] not in validation_ids]
    validation_records = [record for record in records if record["conversation_id"] in validation_ids]
    if not train_records:
        train_records, validation_records = records[:-1], records[-1:]
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
        prompt = f"<|user|>\n{record['user_text']}</s>\n<|assistant|>\n"
        complete = prompt + record["assistant_text"] + "</s>"
        encoded = self.tokenizer(
            complete,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_ids = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, add_special_tokens=True
        )["input_ids"]
        input_ids = encoded["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[: min(len(prompt_ids), self.max_length)] = -100
        labels[encoded["attention_mask"].squeeze(0) == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }


def validation_loss(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            total += model(**batch).loss.item()
    return total / max(len(loader), 1)


def train(args):
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
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        result = {
            "epoch": epoch + 1,
            "train_loss": total / max(len(train_loader), 1),
            "validation_loss": validation_loss(model, validation_loader, device),
        }
        history.append(result)
        print(json.dumps(result))

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metrics = {
        "base_model": args.base_model,
        "algorithm": "LoRA supervised causal language-model fine-tuning",
        "records": len(records),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
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
    parser.add_argument("--minimum-records", type=int, default=100)
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
