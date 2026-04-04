"""
PHASE 3: PAIRWISE FEATURE COMPUTATION
Compute 27 features describing the relationship between every pair of queries.

Two modes:
  train  - pairs from (Role='train') users, with labels
             label 1: both queries from same real user  (same source)
             label 0: one real query + one fake query   (different source)
           Output: output_dir/train_pairs.pkl

  target - all pairs within the target user's observed stream SO (Role='target')
           No labels used — adversary does not know which are real vs fake.
           Output: output_dir/target_pairs.pkl

Input : output_dir/query_features.pkl  (from phase 2)
"""

import argparse
import os
import pickle
import random
from collections import defaultdict
from itertools import combinations
import numpy as np
from tqdm import tqdm
import Levenshtein # type: ignore


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Pairwise Feature Computation")
    parser.add_argument("--features", required=True)
    parser.add_argument("--mode", required=True, choices=["train", "target"])
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--pairs_per_user", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# Similarity helpers
def jaccard(set_a, set_b):
    """Jaccard coefficient between two sets. Returns 0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def cosine_sim(a, b):
    """Cosine similarity between two numpy vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# Core: compute 27 pairwise features for one (qi, qj) pair
def compute_pairwise_features(qi, qj):
    """
    Given two query feature dicts (from phase 2), return a dict of
    27 pairwise similarity / difference features.

    Naming convention mirrors the paper:
        D_*  = absolute difference between the two values
        S_*  = similarity or shared-boolean between the two values
    """
    feats = {}

    # Temporal difference
    feats["D_Time"] = abs(qi["timestamp"] - qj["timestamp"])

    # Same weekday-vs-weekend category
    feats["S_WeekWeekend"] = int(qi["is_weekend"] == qj["is_weekend"])

    # Same 2-hour window of the day
    feats["S_SameDaytime"] = int(
        (qi["hour_of_day"] // 2) == (qj["hour_of_day"] // 2)
    )

    # Click count difference
    feats["D_NumberClicks"] = abs(qi["num_clicks"] - qj["num_clicks"])

    # Jaccard similarity of query term sets
    terms_i = set(str(qi["Query"]).lower().split())
    terms_j = set(str(qj["Query"]).lower().split())
    feats["S_QueryTerms"] = jaccard(terms_i, terms_j)

    # Word count difference
    feats["D_QueryTermLen"] = abs(qi["num_terms"] - qj["num_terms"])

    # Character length difference
    feats["D_QueryCharLen"] = abs(qi["num_chars"] - qj["num_chars"])

    # Levenshtein edit distance between raw query strings -- Small distance → user refining a previous query in the same session
    feats["D_EditDistance"] = Levenshtein.distance(
        str(qi["Query"]).lower(),
        str(qj["Query"]).lower()
    )

    # Term popularity (corpus frequency) difference
    feats["D_QueryTermWeight"] = abs(qi["term_weight"] - qj["term_weight"])

    # Spelling error flags -- Both queries contain at least one spelling error
    feats["S_SpellingError1"] = int(
        qi["has_spelling_error"] == 1 and qj["has_spelling_error"] == 1
    )
    # Boolean difference (one has errors, other does not)
    feats["S_SpellingError2"] = int(
        qi["has_spelling_error"] != qj["has_spelling_error"]
    )
    # Absolute count difference
    feats["D_SpellingErrors"] = abs(
        qi["num_spelling_errors"] - qj["num_spelling_errors"]
    )

    # Location flags -- Both mention the exact same city name
    feats["S_City"] = int(
        qi["city_name"] != "" and qi["city_name"] == qj["city_name"]
    )
    # Both mention the exact same country name
    feats["S_Country"] = int(
        qi["country_name"] != "" and qi["country_name"] == qj["country_name"]
    )
    # Both queries contain any location term
    feats["S_Location1"] = int(
        qi["has_location"] == 1 and qj["has_location"] == 1
    )
    # One has a location term, other does not
    feats["S_Location2"] = int(
        qi["has_location"] != qj["has_location"]
    )

    # SBERT semantic similarity
    # Replaces S_Level2Cat and D_TreeDistance (ODP-based features) from the paper.
    # Cosine similarity between 384-d sentence embeddings captures semantic
    # closeness without requiring an external topic taxonomy.
    feats["S_SemanticSimilarity"] = cosine_sim(qi["embedding"], qj["embedding"])

    return feats


# Convert list of feature dicts -> numpy matrix
def dicts_to_matrix(pair_dicts):
    """
    Convert a list of pairwise feature dicts into a numpy array.
    Returns (X: np.ndarray shape [N, 27], feature_names: list[str]).
    Feature names are sorted alphabetically for reproducibility.
    """
    if not pair_dicts:
        return np.empty((0, 0)), []
    feature_names = sorted(pair_dicts[0].keys())
    X = np.array([[d[k] for k in feature_names] for d in pair_dicts],
                 dtype=np.float32)
    return X, feature_names


# Train mode: labeled pairs from held-out users
def build_train_pairs(features_list, pairs_per_user, seed):
    """
    Build labeled (qi, qj) pairs from training users.

    For each training user:
      - Sample up to `pairs_per_user` same-real-user pairs  → label 1
      - Sample up to `pairs_per_user` real-vs-fake pairs    → label 0

    Why sample and not use all pairs?
    A user with 800 queries would produce 800*799/2 ≈ 320 000 pairs.
    With 100 users that is ~32 M pairs — unmanageable for training.
    Sampling keeps class balance and memory in check.
    """
    random.seed(seed)

    # Group queries by user and label
    user_real = defaultdict(list)
    user_fake = defaultdict(list)
    for f in features_list:
        if f["Role"] != "train":
            continue
        if f["Label"] == "real":
            user_real[f["AnonID"]].append(f)
        elif f["Label"] == "fake":
            user_fake[f["AnonID"]].append(f)

    all_pairs = []
    all_labels = []

    for user in tqdm(sorted(user_real.keys()), desc="  Building training pairs"):
        reals = user_real[user]
        fakes = user_fake[user]

        if len(reals) < 2:
            continue

        # Label-1 pairs: both queries from the same real user
        pos_pool = list(combinations(range(len(reals)), 2))
        random.shuffle(pos_pool)
        for i, j in pos_pool[:pairs_per_user]:
            pf = compute_pairwise_features(reals[i], reals[j])
            all_pairs.append(pf)
            all_labels.append(1)

        # Label-0 pairs: one real query + one fake query
        if fakes:
            neg_pool = [(ri, fi)
                        for ri in range(len(reals))
                        for fi in range(len(fakes))]
            random.shuffle(neg_pool)
            for ri, fi in neg_pool[:pairs_per_user]:
                pf = compute_pairwise_features(reals[ri], fakes[fi])
                all_pairs.append(pf)
                all_labels.append(0)

    return all_pairs, all_labels


# Target mode: all pairs within the observed stream SO
def build_target_pairs(features_list):
    """
    STEP 32: Build every (qi, qj) combination within the target user's
    observed stream SO.

    No labels are used here — the adversary does not know which queries are
    real vs fake. Labels are retained in target_queries only so that
    phase 5 can compute accuracy metrics after the attack.

    Returns:
        pair_dicts    - list of 27-feature dicts, one per pair
        pair_indices  - list of (pos_i, pos_j) ints, position in target_queries
        target_queries - ordered list of query feature dicts in SO
    """
    target_queries = [f for f in features_list if f["Role"] == "target"]

    if not target_queries:
        raise ValueError(
            "No queries with Role='target' found. "
            "Check that your CSV has a 'Role' column with value 'target'."
        )

    n = len(target_queries)
    n_pairs = n * (n - 1) // 2
    print(f"  Target stream: {n} queries → {n_pairs:,} pairs")

    pair_dicts = []
    pair_indices = []

    for i, j in tqdm(
        combinations(range(n), 2),
        total=n_pairs,
        desc="  Computing target pairs"
    ):
        pf = compute_pairwise_features(target_queries[i], target_queries[j])
        pair_dicts.append(pf)
        pair_indices.append((i, j)) # positions in target_queries list

    return pair_dicts, pair_indices, target_queries


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[PHASE 3] Pairwise Feature Computation  (mode={args.mode})")
    print(f"  Features: {args.features}")

    with open(args.features, "rb") as f:
        features_list = pickle.load(f)
    print(f"  Loaded {len(features_list):,} query feature records")

    if args.mode == "train":
        pair_dicts, labels = build_train_pairs(
            features_list, args.pairs_per_user, args.seed
        )
        X, feature_names = dicts_to_matrix(pair_dicts)

        out = {
            "X": X, # shape (N_pairs, 27)
            "y": np.array(labels, dtype=np.int32),
            "feature_names": feature_names,
        }
        out_path = os.path.join(args.output_dir, "train_pairs.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(out, f)

        pos = int(sum(labels))
        neg = len(labels) - pos
        print(f"\n  Saved {len(labels):,} training pairs -> {out_path}")
        print(f"    Positive (same source): {pos:,}")
        print(f"    Negative (real + fake): {neg:,}")

    elif args.mode == "target":
        pair_dicts, pair_indices, target_queries = build_target_pairs(features_list)
        X, feature_names = dicts_to_matrix(pair_dicts)

        out = {
            "X": X, # shape (N_pairs, 27)
            "pair_indices": pair_indices, # list of (pos_i, pos_j)
            "feature_names": feature_names,
            "target_queries": target_queries, # ordered SO queries with metadata
        }
        out_path = os.path.join(args.output_dir, "target_pairs.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(out, f)

        print(f"\n  Saved {len(pair_dicts):,} target pairs -> {out_path}")


if __name__ == "__main__":
    main()
