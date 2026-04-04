"""
PHASE 1: DATA PREPARATION
Converts aol_sample.csv into the base CSV format expected by the pipeline.

What this script does:
  1. Filters to users with enough queries
  2. Selects one user as the attack target (Role='target')
  3. Assigns Role='train' to all other users
  4. Marks every row as Label='real'
  5. Drops SessionID (not needed downstream)

What this script does NOT do:
  - Generate fake queries  → handled by the obfuscation team (Hidden in Plain Sight)
  - Apply DP-COMET         → handled by the obfuscation team

The obfuscation team takes this file as input, adds their fake rows with
Label='fake', and outputs the final pipeline_ready.csv for phases 2-5.

Input : aol_sample.csv  (AnonID, Query, QueryTime, ItemRank, ClickURL, SessionID)
Output: pipeline_ready.csv  (same rows + Label='real' + Role='train'/'target')
"""

import argparse
import os
import numpy as np
import pandas as pd


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Data Preparation")
    p.add_argument("--input", default="aol_sample.csv")
    p.add_argument("--output", default="pipeline_ready.csv")
    p.add_argument("--target_user", type=int, default=None)
    p.add_argument("--min_queries", type=int, default=20)
    p.add_argument("--max_train_users", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Main
def prepare(args):
    print(f"\n[PHASE 0] Data Preparation")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}\n")

    # load
    df = pd.read_csv(args.input)
    df["QueryTime"] = pd.to_datetime(df["QueryTime"])
    df["Query"] = df["Query"].fillna("").astype(str)
    df["ClickURL"] = df.get("ClickURL", pd.Series("", index=df.index)).fillna("")
    df["ItemRank"] = df.get("ItemRank", pd.Series("", index=df.index)).fillna("")
    df = df.drop(columns=["SessionID"], errors="ignore")

    print(f"  Loaded {len(df):,} rows | {df['AnonID'].nunique():,} users")

    # filter users with too few queries
    query_counts = df.groupby("AnonID").size()
    eligible = query_counts[query_counts >= args.min_queries].index
    df = df[df["AnonID"].isin(eligible)]
    print(f"  After min_queries={args.min_queries} filter: "
          f"{df['AnonID'].nunique():,} users | {len(df):,} rows")

    if df["AnonID"].nunique() < 2:
        raise ValueError(
            f"Fewer than 2 users remain. Lower --min_queries (currently {args.min_queries})."
        )

    # select target user
    if args.target_user is not None:
        if args.target_user not in df["AnonID"].values:
            raise ValueError(f"--target_user {args.target_user} not found after filtering.")
        target_id = args.target_user
    else:
        target_id = int(query_counts[eligible].idxmax())

    print(f"  Target user: AnonID={target_id} "
          f"({int((df['AnonID'] == target_id).sum())} queries)")

    # cap training users if requested
    train_ids = [uid for uid in eligible if uid != target_id]
    if args.max_train_users is not None and len(train_ids) > args.max_train_users:
        rng = np.random.default_rng(args.seed)
        train_ids = list(rng.choice(train_ids, size=args.max_train_users, replace=False))
        print(f"  Capped training users to {args.max_train_users}")

    all_ids = train_ids + [target_id]
    df = df[df["AnonID"].isin(all_ids)].copy()

    # assign Label and Role
    df["Label"] = "real"
    df["Role"] = df["AnonID"].apply(lambda uid: "target" if uid == target_id else "train")

    # sort chronologically
    df = df.sort_values(["AnonID", "QueryTime"]).reset_index(drop=True)

    # summary
    print(f"\n  Output summary:")
    print(f"    Total rows  : {len(df):,}")
    print(f"    All Label   : real  (fakes added by obfuscation team)")
    print(f"    Target rows : {(df['Role']=='target').sum():,}  (AnonID={target_id})")
    print(f"    Train users : {df[df['Role']=='train']['AnonID'].nunique():,}")

    # write
    df.to_csv(args.output, index=False)
    print(f"\n  Wrote: {args.output}")
    print(f"\n  Next step: obfuscation team adds fake rows with Label='fake'")
    print(f"  Then run the attack pipeline with: make all")


if __name__ == "__main__":
    args = parse_args()
    prepare(args)