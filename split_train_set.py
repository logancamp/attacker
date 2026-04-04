"""
split_train_set.py

Splits pipeline_ready.csv into two separate files:
  - training set  (Role='train')  → used by phases 3-4 to train GBRT
  - target set    (Role='target') → used by phases 3, 5 for the attack

This keeps the files manageable and makes the pipeline explicit about
what data each phase is consuming.

Input : pipeline_ready.csv
Output: data/train_set.csv   (all Role='train' rows)
        data/target_set.csv  (all Role='target' rows)
"""

import argparse
import pandas as pd
import os


def parse_args():
    p = argparse.ArgumentParser(description="Split pipeline_ready.csv into train/target")
    p.add_argument("--input",       default="pipeline_ready.csv")
    p.add_argument("--train_out",   default="data/train_set.csv")
    p.add_argument("--target_out",  default="data/target_set.csv")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    print(f"\n  Loaded {len(df):,} rows from {args.input}")

    train  = df[df["Role"] == "train"].reset_index(drop=True)
    target = df[df["Role"] == "target"].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.train_out),  exist_ok=True)
    os.makedirs(os.path.dirname(args.target_out), exist_ok=True)

    train.to_csv(args.train_out,   index=False)
    target.to_csv(args.target_out, index=False)

    print(f"\n  Train  set: {len(train):,} rows  "
          f"({train['AnonID'].nunique()} users, "
          f"{(train['Label']=='real').sum()} real, "
          f"{(train['Label']=='fake').sum()} fake) "
          f"-> {args.train_out}")

    print(f"  Target set: {len(target):,} rows  "
          f"({(target['Label']=='real').sum()} real, "
          f"{(target['Label']=='fake').sum()} fake) "
          f"-> {args.target_out}")


if __name__ == "__main__":
    main()