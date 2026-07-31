"""Stage 1 bootstrap: train the nuance_encoder + classifier head on the
existing labeled dataset (data/processed/{train,eval}.parquet, produced by
scripts/prepare_dataset.py) and report baseline metrics, with particular
attention to the Severe/Emergency boundary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from inhouse_model import config
from inhouse_model.model import InHouseModel


def main() -> None:
    train_df = pd.read_parquet(config.TRAIN_PARQUET)
    eval_df = pd.read_parquet(config.EVAL_PARQUET)

    model = InHouseModel()
    metadata = model.train(train_df, eval_dataset=eval_df)

    print("\n=== Stage 1 training complete ===")
    print(f"checkpoint version: v{metadata['version']}")
    print(f"train rows: {metadata['num_train_rows']}")


if __name__ == "__main__":
    main()
