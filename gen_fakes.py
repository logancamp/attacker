"""
generate_placeholder_fakes.py

Adds obviously fake queries to pipeline_ready.csv so the full attack
pipeline can be tested end-to-end before real obfuscation is implemented.

Fake queries are random nonsense (random words from a fixed vocabulary)
that no real user would ever type. The attack should separate these
perfectly, giving you a best-case baseline for attack performance.

Input : pipeline_ready_base.csv  (real queries only, never modified)
Output: pipeline_ready.csv        (real + fake, used by phases 2-5)
"""

import argparse
import random
import pandas as pd

# Nonsense vocabulary — clearly not real searches
FAKE_WORDS = [
    "zxqvb", "qqqqq", "asdfgh", "xkqzwp", "bbbbb", "zzzzz",
    "qwerty", "lmnop", "vvvvv", "xyzxyz", "jjjjj", "fffff",
    "ppppp", "nnnnn", "mmmmm", "hhhhh", "ggggg", "ddddd",
]

def random_fake_query(rng):
    n_words = rng.randint(1, 3)
    return " ".join(rng.choices(FAKE_WORDS, k=n_words))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="pipeline_ready_base.csv")
    p.add_argument("--output", default="pipeline_ready.csv")
    p.add_argument("--ratio",  type=float, default=1.0,
                   help="Fake queries per real query (default: 1.0)")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    df = pd.read_csv(args.input)
    df["QueryTime"] = pd.to_datetime(df["QueryTime"])

    # Guard: refuse to run if input already contains fake queries
    if "Label" in df.columns and (df["Label"] == "fake").any():
        raise ValueError(
            f"{args.input} already contains fake queries. "
            f"Make sure --input points to the real-only base file."
        )

    fake_rows = []

    for uid, group in df.groupby("AnonID"):
        role     = group["Role"].iloc[0]
        n_fakes  = max(1, int(len(group) * args.ratio))
        t_min    = group["QueryTime"].min()
        t_max    = group["QueryTime"].max()
        span     = max(1, int((t_max - t_min).total_seconds()))

        for _ in range(n_fakes):
            offset = rng.randint(0, span)
            fake_rows.append({
                "AnonID":    uid,
                "Query":     random_fake_query(rng),
                "QueryTime": t_min + pd.Timedelta(seconds=offset),
                "ItemRank":  "",
                "ClickURL":  "",
                "Label":     "fake",
                "Role":      role,
            })

    fakes_df = pd.DataFrame(fake_rows)
    out      = pd.concat([df, fakes_df], ignore_index=True)
    out      = out.sort_values(["AnonID", "QueryTime"]).reset_index(drop=True)
    out.to_csv(args.output, index=False)

    print(f"Added {len(fakes_df):,} fake rows -> {args.output}")
    print(f"  Real : {(out['Label']=='real').sum():,}")
    print(f"  Fake : {(out['Label']=='fake').sum():,}")


if __name__ == "__main__":
    main()