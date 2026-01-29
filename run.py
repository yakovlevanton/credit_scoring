from __future__ import annotations

import argparse

from src.train import train_and_save
from src.test import predict_and_save


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/raw")
    p.add_argument("--model-path", type=str, default="models/catboost.cbm")
    p.add_argument("--out", type=str, default="predictions/submission.csv")
    p.add_argument("--mode", choices=["train", "predict"], default="train")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "train":
        train_and_save(args.data_dir, args.model_path, random_state=args.seed)
    else:
        predict_and_save(args.data_dir, args.model_path, args.out)


if __name__ == "__main__":
    main()

