#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Set
import numpy as np
import pandas as pd


# Core sampling
@dataclass
class SampleConfig:
    input_path: str
    output_csv: str
    output_queries_only_csv: Optional[str]
    seed: int
    target_users: int
    max_queries_per_user: int
    session_gap_minutes: Optional[int]


def read_full_file(path: str) -> pd.DataFrame:
    cols = ["AnonID", "Query", "QueryTime", "ItemRank", "ClickURL"]
    return pd.read_csv(
        path,
        sep="\t",
        names=cols,
        header=0,
        dtype={"AnonID": "Int64", "Query": "string", "QueryTime": "string",
               "ItemRank": "string", "ClickURL": "string"},
        engine="python",
        on_bad_lines="skip",
        keep_default_na=False,
        quoting=3,  # csv.QUOTE_NONE
    )


def minimal_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows missing required fields
    df = df[df["AnonID"].notna()].copy()

    # Normalize Query
    df["Query"] = df["Query"].astype("string").str.strip()
    df = df[df["Query"].notna() & (df["Query"].str.len() > 0)]

    # Keep queries with at least 2 chars
    df = df[df["Query"].str.len() >= 2]

    # Convert time string to datetime, drop rows with invalid times
    df["QueryTime"] = pd.to_datetime(df["QueryTime"], errors="coerce")
    df = df[df["QueryTime"].notna()]

    # Deduplicate click duplicates: identical (AnonID, Query, QueryTime)
    df = df.drop_duplicates(subset=["AnonID", "Query", "QueryTime"], keep="first")

    return df


def pick_users(df: pd.DataFrame, cfg: SampleConfig) -> Set[int]:
    rng = np.random.default_rng(cfg.seed)
    user_pool = df["AnonID"].dropna().astype(int).unique().tolist()

    if len(user_pool) < cfg.target_users:
        raise ValueError(
            f"User pool too small ({len(user_pool)}) after cleaning. "
            f"Decrease --target_users."
        )

    chosen = rng.choice(np.array(user_pool), size=cfg.target_users, replace=False)
    return set(map(int, chosen))


def sessionize(df: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    df = df.sort_values(["AnonID", "QueryTime"]).copy()
    # time difference within each user
    dt = df.groupby("AnonID")["QueryTime"].diff()
    new_session = (dt.isna()) | (dt > pd.Timedelta(minutes=gap_minutes))
    # cumulative session count per user
    df["SessionID"] = new_session.groupby(df["AnonID"]).cumsum().astype(int)
    return df


def sample_queries(df: pd.DataFrame, cfg: SampleConfig, chosen_users: Set[int]) -> pd.DataFrame:
    df = df[df["AnonID"].astype(int).isin(chosen_users)]

    if df.empty:
        raise RuntimeError("No rows collected for chosen users. Check parsing / delimiter / file path.")

    df = df.sort_values(["AnonID", "QueryTime"]).reset_index(drop=True)

    # Cap queries per user
    df["row_in_user"] = df.groupby("AnonID").cumcount()
    df = df[df["row_in_user"] < cfg.max_queries_per_user].drop(columns=["row_in_user"])

    if cfg.session_gap_minutes is not None:
        df = sessionize(df, cfg.session_gap_minutes)

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True, help="Path to user-ct-test-collection-02.txt")
    p.add_argument("--output_csv", default="aol_sample.csv", help="Output sample CSV path")
    p.add_argument("--output_queries_only_csv", default="aol_queries_only.csv",
                   help="Optional: output queries-only CSV (AnonID, Query, QueryTime, SessionID if enabled)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_users", type=int, default=1000, help="How many users to sample")
    p.add_argument("--max_queries_per_user", type=int, default=100, help="Cap queries per user")
    p.add_argument("--session_gap_minutes", type=int, default=30,
                   help="Session gap threshold in minutes (e.g., 30)")
    args = p.parse_args()

    cfg = SampleConfig(
        input_path=args.input_path,
        output_csv=args.output_csv,
        output_queries_only_csv=args.output_queries_only_csv if args.output_queries_only_csv else None,
        seed=args.seed,
        target_users=args.target_users,
        max_queries_per_user=args.max_queries_per_user,
        session_gap_minutes=args.session_gap_minutes,
    )

    if not os.path.exists(cfg.input_path):
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")

    print("Reading full file and cleaning...")
    df = read_full_file(cfg.input_path)
    df = minimal_clean(df)
    print(f"Cleaned rows: {len(df):,}")

    print("Picking users...")
    chosen_users = pick_users(df, cfg)
    print(f"Chosen users: {len(chosen_users):,}")

    print("Sampling queries for chosen users...")
    df_sample = sample_queries(df, cfg, chosen_users)
    print(f"Sample rows: {len(df_sample):,}")
    print(f"Unique users in sample: {df_sample['AnonID'].nunique():,}")

    # Save full sample (keeps ItemRank/ClickURL if present)
    df_sample.to_csv(cfg.output_csv, index=False)
    print(f"Wrote: {cfg.output_csv}")

    # Save queries-only (useful for DP-COMET)
    if cfg.output_queries_only_csv:
        keep_cols = ["AnonID", "Query", "QueryTime"]
        if "SessionID" in df_sample.columns:
            keep_cols.append("SessionID")
        df_sample[keep_cols].to_csv(cfg.output_queries_only_csv, index=False)
        print(f"Wrote: {cfg.output_queries_only_csv}")

if __name__ == "__main__":
    main()
