"""
temp_clean.py

Converts the Random Query Injection output CSV from the obfuscation code
into the format expected by phases 2-5.

Problems this fixes (skipped if already done):
  1. Fake rows have null AnonID -> filled from the real row in the same group_id
  2. Fake rows have null QueryTime -> assigned from the paired real row's timestamp
  3. Column 'label' renamed to 'Label'
  4. 'Role' column added
  5. Extra columns dropped (language, ratio_setting, group_id, word_count, SessionID)
"""

import argparse
import random
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Clean RQI output for attack pipeline")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="pipeline_ready.csv")
    p.add_argument("--target_user", type=int, default=None)
    p.add_argument("--min_queries", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    print(f"\n[CLEAN RQI] {args.input} -> {args.output}\n")

    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df):,} rows | Columns: {list(df.columns)}")

    # rename label -> Label if needed
    if "label" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={"label": "Label"})
        print("  Renamed 'label' -> 'Label'")
    elif "Label" in df.columns:
        print("  'Label' column already present, skipping rename")

    # fill AnonID for fake rows if needed
    if "group_id" in df.columns and df["AnonID"].isna().any():
        print("  Filling null AnonIDs from group_id...")
        group_to_user = (
            df[df["Label"] == "real"]
            .groupby("group_id")["AnonID"]
            .first()
            .to_dict()
        )
        fake_mask = df["Label"] == "fake"
        df.loc[fake_mask, "AnonID"] = df.loc[fake_mask, "group_id"].map(group_to_user)
        still_null = df["AnonID"].isna().sum()
        if still_null > 0:
            print(f"  WARNING: {still_null} rows still null after fill, dropping them")
            df = df.dropna(subset=["AnonID"])
    else:
        print("  AnonID already populated, skipping fill")

    df["AnonID"] = df["AnonID"].astype(int)

    # fill QueryTime for fake rows if needed
    df["QueryTime"] = pd.to_datetime(df["QueryTime"], errors="coerce")
    if "group_id" in df.columns and df["QueryTime"].isna().any():
        print("  Filling null QueryTimes from group_id...")
        group_to_time = (
            df[df["Label"] == "real"]
            .groupby("group_id")["QueryTime"]
            .first()
            .to_dict()
        )
        fake_mask = df["Label"] == "fake"
        for idx in df[fake_mask & df["QueryTime"].isna()].index:
            gid = df.at[idx, "group_id"]
            base_t = group_to_time.get(gid)
            if base_t:
                df.at[idx, "QueryTime"] = pd.to_datetime(base_t) + \
                    pd.Timedelta(seconds=rng.randint(1, 300))
    else:
        print("  QueryTime already populated, skipping fill")

    # drop extra columns
    drop_cols = ["SessionID", "language", "ratio_setting", "group_id", "word_count"]
    present = [c for c in drop_cols if c in df.columns]
    if present:
        df = df.drop(columns=present)
        print(f"  Dropped extra columns: {present}")

    # clean up
    df["Query"] = df["Query"].fillna("").astype(str)
    df["ClickURL"] = df.get("ClickURL", pd.Series("", index=df.index)).fillna("")
    df["ItemRank"] = df.get("ItemRank", pd.Series("", index=df.index)).fillna("")

    # filter users with too few real queries
    real_counts = df[df["Label"] == "real"].groupby("AnonID").size()
    eligible = real_counts[real_counts >= args.min_queries].index
    df = df[df["AnonID"].isin(eligible)]
    print(f"  After min_queries={args.min_queries} filter: "
          f"{df['AnonID'].nunique():,} users | {len(df):,} rows")

    # assign Role if needed
    if "Role" in df.columns:
        print("  'Role' column already present, skipping assignment")
    else:
        if args.target_user is not None:
            if args.target_user not in df["AnonID"].values:
                raise ValueError(f"--target_user {args.target_user} not found.")
            target_id = args.target_user
        else:
            target_id = int(real_counts[eligible].idxmax())
        print(f"  Assigning Role — target: AnonID={target_id}")
        df["Role"] = df["AnonID"].apply(
            lambda uid: "target" if uid == target_id else "train"
        )

    # sort and validate
    df = df.sort_values(["AnonID", "QueryTime"]).reset_index(drop=True)

    required = ["AnonID", "Query", "QueryTime", "Label", "Role"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Output is still missing required columns: {missing}")

    print(f"\n  Output summary:")
    print(f"    Total rows  : {len(df):,}")
    print(f"    Real queries: {(df['Label']=='real').sum():,}")
    print(f"    Fake queries: {(df['Label']=='fake').sum():,}")
    print(f"    Target rows : {(df['Role']=='target').sum():,}")
    print(f"    Train users : {df[df['Role']=='train']['AnonID'].nunique():,}")

    df.to_csv(args.output, index=False)
    print(f"\n  Wrote: {args.output}")


if __name__ == "__main__":
    main()