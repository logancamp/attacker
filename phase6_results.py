"""
PHASE 6: RESULTS SUMMARY
Loads attack outputs and produces human-readable metrics and topic profiles.

What this computes:
  - Query privacy:   FP/FN rates from the clustering attack
  - Semantic privacy: cosine distance between adversary's inferred topic
                      profile and the user's true topic profile
  - Relative privacy: how much the obfuscation actually helped vs baseline
  - Topic profiles:  zero-shot classification of queries into 10 topics
                     showing exactly what the adversary learned about the user

Input : output/cluster_results.csv   (from phase 5)
        output/attack_metrics.pkl    (from phase 5)
        pipeline_ready.csv           (original data for baseline)

Output: output/results_summary.txt   human-readable report
        output/topic_profiles.csv    true vs inferred profile per topic
        output/metrics_table.csv     all numeric metrics in one table
        (printed to terminal as well)

Note: Zero-shot classification requires transformers package.
      Install with: pip install transformers
      First run downloads facebook/bart-large-mnli (~1.6 GB).
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

TOPICS = [
    "health and medicine",
    "finance and money",
    "sports and fitness",
    "technology and computers",
    "food and restaurants",
    "travel and places",
    "entertainment and media",
    "education and learning",
    "shopping and products",
    "news and politics",
]


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Phase 6: Results Summary")
    p.add_argument("--cluster_results", default="output/cluster_results.csv")
    p.add_argument("--attack_metrics",  default="output/attack_metrics.pkl")
    p.add_argument("--output_dir",      default="output")
    p.add_argument("--no_profiles",     action="store_true",
                   help="Skip zero-shot topic profiling (faster, no transformers needed)")
    return p.parse_args()


# =============================================================================
# Topic profiling via zero-shot classification
# =============================================================================

def load_classifier():
    """Load zero-shot classification pipeline."""
    try:
        from transformers import pipeline
        print("  Loading zero-shot classifier (downloads ~1.6 GB on first run)...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU
        )
        return classifier
    except ImportError:
        raise ImportError(
            "transformers package not found. "
            "Install with: pip install transformers\n"
            "Or run with --no_profiles to skip topic profiling."
        )


def classify_queries(queries, classifier, batch_size=32):
    """
    Run zero-shot classification on a list of query strings.
    Returns numpy array of shape (n_queries, n_topics).
    """
    all_scores = []
    queries = [str(q).strip() for q in queries]

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        # Filter empty strings
        batch = [q if q else "unknown" for q in batch]
        results = classifier(batch, TOPICS, multi_label=True)
        if isinstance(results, dict):
            results = [results]
        for r in results:
            # Re-order scores to match TOPICS list order
            score_dict = dict(zip(r["labels"], r["scores"]))
            scores = [score_dict.get(t, 0.0) for t in TOPICS]
            all_scores.append(scores)

    return np.array(all_scores)


def build_profile(score_matrix):
    """Average topic scores across all queries to get a profile vector."""
    if len(score_matrix) == 0:
        return np.zeros(len(TOPICS))
    return score_matrix.mean(axis=0)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[PHASE 6] Results Summary\n")

    # ---- load cluster results ----
    results_df = pd.read_csv(args.cluster_results)
    print(f"  Loaded {len(results_df)} query assignments from {args.cluster_results}")

    # ---- load attack metrics ----
    with open(args.attack_metrics, "rb") as f:
        metrics = pickle.load(f)

    # ---- separate real and predicted-real queries ----
    true_real      = results_df[results_df["TrueLabel"] == "real"]
    predicted_real = results_df[results_df["PredictedLabel"] == "real"]
    predicted_fake = results_df[results_df["PredictedLabel"] == "fake"]

    # ---- query privacy metrics ----
    fp_rate        = metrics["fp_rate"]
    fn_rate        = metrics["fn_rate"]
    attack_acc     = metrics["attack_accuracy"]

    print(f"\n  === QUERY PRIVACY METRICS ===")
    print(f"  False Positive Rate : {fp_rate:.4f}  (real+fake merged — higher = better privacy)")
    print(f"  False Negative Rate : {fn_rate:.4f}  (real+real split  — higher = better privacy)")
    print(f"  Attack Accuracy     : {attack_acc:.4f}  (lower = better privacy for user)")

    print(f"\n  Cluster composition:")
    for label in ["real", "fake"]:
        cluster = results_df[results_df["PredictedLabel"] == label]
        n_correct = (cluster["TrueLabel"] == label).sum()
        n_wrong   = (cluster["TrueLabel"] != label).sum()
        print(f"    Predicted {label:4s}: {len(cluster):4d} queries "
              f"({n_correct} correct, {n_wrong} wrong)")

    # ---- topic profiles ----
    semantic_privacy = None
    profile_df       = None

    if not args.no_profiles:
        print(f"\n  === TOPIC PROFILES ===")
        classifier = load_classifier()

        # True profile: built from ground-truth real queries
        print(f"  Classifying {len(true_real)} true real queries...")
        true_scores   = classify_queries(true_real["Query"].tolist(), classifier)
        true_profile  = build_profile(true_scores)

        # Inferred profile: built from what adversary labeled as real
        print(f"  Classifying {len(predicted_real)} adversary-predicted real queries...")
        infer_scores   = classify_queries(predicted_real["Query"].tolist(), classifier)
        infer_profile  = build_profile(infer_scores)

        # Semantic privacy = cosine distance between profiles
        # Distance of 0 = identical profiles = no privacy
        # Distance of 1 = completely different profiles = perfect privacy
        cos_sim          = float(cosine_similarity([true_profile], [infer_profile])[0][0])
        semantic_privacy = 1.0 - cos_sim

        print(f"\n  Semantic Privacy Score : {semantic_privacy:.4f}  "
              f"(cosine distance — higher = better privacy)")
        print(f"  (0 = adversary has perfect profile, 1 = adversary is completely wrong)")

        # Build profile comparison table
        profile_df = pd.DataFrame({
            "Topic":          TOPICS,
            "TrueProfile":    np.round(true_profile,  4),
            "InferredProfile": np.round(infer_profile, 4),
            "Difference":     np.round(np.abs(true_profile - infer_profile), 4),
        }).sort_values("TrueProfile", ascending=False).reset_index(drop=True)

        print(f"\n  Topic profile comparison (true vs adversary inferred):")
        print(f"  {'Topic':<30s} {'True':>8s}  {'Inferred':>8s}  {'Diff':>8s}")
        print(f"  {'-'*58}")
        for _, row in profile_df.iterrows():
            print(f"  {row['Topic']:<30s} {row['TrueProfile']:>8.4f}  "
                  f"{row['InferredProfile']:>8.4f}  {row['Difference']:>8.4f}")

    # ---- metrics table ----
    metrics_row = {
        "n_queries":        metrics["n_queries"],
        "n_real":           len(true_real),
        "n_fake":           len(results_df) - len(true_real),
        "fp_rate":          round(fp_rate, 4),
        "fn_rate":          round(fn_rate, 4),
        "attack_accuracy":  round(attack_acc, 4),
        "semantic_privacy": round(semantic_privacy, 4) if semantic_privacy is not None else "n/a",
    }
    metrics_table = pd.DataFrame([metrics_row])

    # ---- save outputs ----
    metrics_path = os.path.join(args.output_dir, "metrics_table.csv")
    metrics_table.to_csv(metrics_path, index=False)
    print(f"\n  Saved metrics table    -> {metrics_path}")

    if profile_df is not None:
        profile_path = os.path.join(args.output_dir, "topic_profiles.csv")
        profile_df.to_csv(profile_path, index=False)
        print(f"  Saved topic profiles   -> {profile_path}")

    # ---- write text report ----
    report_path = os.path.join(args.output_dir, "results_summary.txt")
    with open(report_path, "w") as f:
        f.write("LINKAGE ATTACK RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target stream:       {metrics['n_queries']} queries\n")
        f.write(f"Real queries:        {len(true_real)}\n")
        f.write(f"Fake queries:        {len(results_df) - len(true_real)}\n\n")
        f.write("QUERY PRIVACY\n")
        f.write("-" * 30 + "\n")
        f.write(f"False Positive Rate: {fp_rate:.4f}\n")
        f.write(f"False Negative Rate: {fn_rate:.4f}\n")
        f.write(f"Attack Accuracy:     {attack_acc:.4f}\n\n")
        if semantic_privacy is not None:
            f.write("SEMANTIC PRIVACY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Semantic Privacy:    {semantic_privacy:.4f}\n\n")
            f.write("TOPIC PROFILES\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Topic':<30s} {'True':>8s}  {'Inferred':>8s}  {'Diff':>8s}\n")
            for _, row in profile_df.iterrows():
                f.write(f"{row['Topic']:<30s} {row['TrueProfile']:>8.4f}  "
                        f"{row['InferredProfile']:>8.4f}  {row['Difference']:>8.4f}\n")

    print(f"  Saved results report   -> {report_path}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()