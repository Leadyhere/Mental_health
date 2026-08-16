"""Download the gated MentalBERT base model after Hugging Face authentication."""

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "mental/mental-bert-base-uncased"
OUTPUT = Path(__file__).resolve().parent.parent / "models" / "base" / "mental-bert-base-uncased"


def main():
    try:
        snapshot_download(MODEL_ID, local_dir=OUTPUT)
    except Exception as exc:
        raise SystemExit(
            "MentalBERT is gated. Open its Hugging Face page, accept the access terms, "
            "then run `hf auth login` and retry this script.\n\n"
            f"Original error: {exc}"
        ) from exc
    print(f"MentalBERT downloaded to {OUTPUT}")


if __name__ == "__main__":
    main()
