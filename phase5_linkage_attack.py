"""
PHASE 5: LINKAGE ATTACK
Execute the full Gervais linkage attack on the target user's observed stream SO.

Pipeline:
  1. Load all 60 trained GBRT models
  2. Score every pair in SO through each model
  3. Take the median score across all models per pair  (robustness)
  4. Arrange scores into an N×N similarity matrix
  5. Run k-means with k=2 → Cluster 'real' | Cluster 'fake'
  6. Compute query-level and semantic privacy metrics

Input : output_dir/target_pairs.pkl   (from phase 3 target mode)
        output_dir/models/             (from phase 4)
Output: output_dir/cluster_results.csv
        output_dir/similarity_matrix.pkl
        output_dir/attack_metrics.pkl
"""

import argparse
import os
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5: Linkage Attack")
    parser.add_argument("--target_pairs", required=True,
                        help="Path to target_pairs.pkl from phase 3 (target mode)")
    parser.add_argument("--models_dir",   required=True,
                        help="Directory containing model_XX.pkl files from phase 4")
    parser.add_argument("--output_dir",   default="output",
                        help="Directory for output files")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_models(models_dir):
    """Load all GBRT model files from models_dir."""
    model_files = sorted([
        f for f in os.listdir(models_dir)
        if f.startswith("model_") and f.endswith(".pkl")
    ])
    if not model_files:
        raise FileNotFoundError(f"No model_XX.pkl files found in {models_dir}")

    models = []
    for mf in tqdm(model_files, desc="  Loading models"):
        models.append(joblib.load(os.path.join(models_dir, mf)))

    print(f"  Loaded {len(models)} GBRT models")
    return models


# ---------------------------------------------------------------------------
# STEP 33-34: Score pairs through all models, take median
# ---------------------------------------------------------------------------

def score_pairs(X, models):
    """
    STEP 33: Run each pair through every GBRT model.
             Each model outputs P(same_source) in [0, 1].

    STEP 34: Take the median across all models per pair.
             Median is used (not mean) because it is robust to outlier models
             — a single badly-fit model on an unrepresentative subset will not
             drag the aggregate score in the wrong direction.

    Returns: np.ndarray shape (N_pairs,) of median linkage scores.
    """
    # all_scores[model_idx, pair_idx] = P(same_source)
    all_scores = np.zeros((len(models), len(X)), dtype=np.float32)

    for i, model in enumerate(tqdm(models, desc="  Scoring pairs")):
        # predict_proba returns [[P(0), P(1)], ...]
        # column 1 = P(same source) = our linkage score
        all_scores[i] = model.predict_proba(X)[:, 1]

    # STEP 34: median across models dimension
    median_scores = np.median(all_scores, axis=0)   # shape (N_pairs,)
    return median_scores


# ---------------------------------------------------------------------------
# STEP 35: Build N×N similarity matrix
# ---------------------------------------------------------------------------

def build_similarity_matrix(median_scores, pair_indices, n_queries):
    """
    STEP 35: Arrange the per-pair median linkage scores into an N×N matrix.

    pair_indices is a list of (pos_i, pos_j) where pos_i / pos_j are
    0-based positions in the target_queries list (set in phase 3).

    The matrix is symmetric: matrix[i,j] = matrix[j,i] = L(qi, qj).
    Diagonal is set to 1.0 (a query is identical to itself).
    """
    matrix = np.zeros((n_queries, n_queries), dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)

    for score, (i, j) in zip(median_scores, pair_indices):
        matrix[i, j] = score
        matrix[j, i] = score   # symmetric

    return matrix


# ---------------------------------------------------------------------------
# STEP 36: K-means clustering on similarity matrix
# ---------------------------------------------------------------------------

def run_kmeans(similarity_matrix, k=2):
    """
    STEP 36: Cluster the N queries into k=2 groups.

    Each row of the similarity matrix is a query's 'similarity profile' —
    how strongly it links to every other query in SO.
    K-means finds two groups that maximise intra-cluster link strength.

    n_init=10: run 10 random initialisations, keep the best result.
    This reduces sensitivity to initialisation.

    Returns: np.ndarray of cluster labels (0 or 1), shape (N,).
    """
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(similarity_matrix)
    return labels


# ---------------------------------------------------------------------------
# Privacy metrics
# ---------------------------------------------------------------------------

def compute_query_privacy(cluster_labels, true_labels):
    """
    Compute False Positive and False Negative rates for query-level privacy.

    Definitions (from paper Section 2.5.1):
      FP (False Positive):  a real query and a fake query end up in the SAME
                            cluster. The fake query 'passed as real'.
                            High FP = good privacy (obfuscation confused attacker).

      FN (False Negative):  two real queries end up in DIFFERENT clusters.
                            The user's own queries were separated.
                            High FN = good privacy.

    Vectorised implementation: avoids Python loop over all pairs.

    Parameters
    ----------
    cluster_labels : np.ndarray shape (N,)  - adversary's cluster assignments
    true_labels    : np.ndarray shape (N,)  - 1=real, 0=fake (ground truth)

    Returns dict with fp_rate, fn_rate, counts.
    """
    real_mask = true_labels == 1
    fake_mask = true_labels == 0

    real_clusters = cluster_labels[real_mask]   # clusters of real queries
    fake_clusters = cluster_labels[fake_mask]   # clusters of fake queries

    n_real = real_clusters.shape[0]
    n_fake = fake_clusters.shape[0]

    # ---- False Negatives: real+real pairs split across clusters ----
    # Outer comparison: real_clusters[i] != real_clusters[j]
    fn_matrix      = real_clusters[:, None] != real_clusters[None, :]
    fn_count       = int(fn_matrix.sum()) // 2          # symmetric, divide by 2
    total_rr_pairs = n_real * (n_real - 1) // 2
    fn_rate        = fn_count / total_rr_pairs if total_rr_pairs > 0 else 0.0

    # ---- False Positives: real+fake pairs in the same cluster ----
    # Cross comparison: real_clusters[i] == fake_clusters[j]
    fp_matrix      = real_clusters[:, None] == fake_clusters[None, :]
    fp_count       = int(fp_matrix.sum())
    total_rf_pairs = n_real * n_fake
    fp_rate        = fp_count / total_rf_pairs if total_rf_pairs > 0 else 0.0

    return {
        "fp_rate":        fp_rate,
        "fn_rate":        fn_rate,
        "fp_count":       fp_count,
        "fn_count":       fn_count,
        "total_rr_pairs": total_rr_pairs,
        "total_rf_pairs": total_rf_pairs,
    }


def label_clusters(cluster_labels, true_labels):
    """
    Decide which cluster (0 or 1) the adversary would label as 'real'.

    Strategy: whichever cluster contains more real queries is labeled 'real'.
    Ties broken by choosing cluster 0.
    Returns: real_cluster (int 0 or 1), fake_cluster (int 1 or 0).
    """
    n_real_c0 = int(((cluster_labels == 0) & (true_labels == 1)).sum())
    n_real_c1 = int(((cluster_labels == 1) & (true_labels == 1)).sum())
    real_cluster = 0 if n_real_c0 >= n_real_c1 else 1
    fake_cluster = 1 - real_cluster
    return real_cluster, fake_cluster


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[PHASE 5] Linkage Attack")
    print(f"  Target pairs : {args.target_pairs}")
    print(f"  Models dir   : {args.models_dir}")
    print(f"  Output dir   : {args.output_dir}\n")

    # ---- Load inputs ----
    with open(args.target_pairs, "rb") as f:
        data = pickle.load(f)

    X              = data["X"]               # shape (N_pairs, 27)
    pair_indices   = data["pair_indices"]    # list of (pos_i, pos_j)
    target_queries = data["target_queries"]  # ordered list of query feature dicts

    n_queries = len(target_queries)
    n_pairs   = len(X)
    print(f"  Target stream : {n_queries} queries | {n_pairs:,} pairs\n")

    # ---- Load models ----
    models = load_models(args.models_dir)

    # ---- STEP 33-34: Score pairs, take median ----
    print()
    median_scores = score_pairs(X, models)

    # ---- STEP 35: Build similarity matrix ----
    print("\n  Building similarity matrix...")
    sim_matrix = build_similarity_matrix(median_scores, pair_indices, n_queries)
    print(f"  Matrix shape: {sim_matrix.shape}  "
          f"(min={sim_matrix.min():.3f}, max={sim_matrix.max():.3f}, "
          f"mean={sim_matrix.mean():.3f})")

    # ---- STEP 36: K-means k=2 ----
    print("\n  Running k-means (k=2)...")
    cluster_labels = run_kmeans(sim_matrix, k=2)

    # ---- Ground-truth labels (for evaluation only) ----
    true_labels = np.array(
        [1 if q["Label"] == "real" else 0 for q in target_queries],
        dtype=np.int32
    )

    # ---- Decide which cluster = real ----
    real_cluster, fake_cluster = label_clusters(cluster_labels, true_labels)

    # ---- Query privacy metrics ----
    qp = compute_query_privacy(cluster_labels, true_labels)

    # ---- Build results dataframe ----
    predicted_labels = np.where(cluster_labels == real_cluster, "real", "fake")
    results_df = pd.DataFrame({
        "query_id":        [q["query_id"]  for q in target_queries],
        "AnonID":          [q["AnonID"]    for q in target_queries],
        "Query":           [q["Query"]     for q in target_queries],
        "QueryTime":       [q["QueryTime"] for q in target_queries],
        "TrueLabel":       [q["Label"]     for q in target_queries],
        "ClusterLabel":    cluster_labels,
        "PredictedLabel":  predicted_labels,
        "Correct":         predicted_labels == np.array(
                               [q["Label"] for q in target_queries]
                           ),
    })

    # ---- Print results ----
    print(f"\n{'='*50}")
    print(f"  ATTACK RESULTS")
    print(f"{'='*50}")
    print(f"\n  Cluster composition:")
    for c in [0, 1]:
        tag    = "REAL" if c == real_cluster else "FAKE"
        n_r    = int(((cluster_labels == c) & (true_labels == 1)).sum())
        n_f    = int(((cluster_labels == c) & (true_labels == 0)).sum())
        total  = n_r + n_f
        print(f"    Cluster {c} (→ labeled {tag}): "
              f"{n_r} real + {n_f} fake = {total} total")

    print(f"\n  Query-level privacy  (higher = better privacy for user):")
    print(f"    False Positive rate: {qp['fp_rate']:.4f}  "
          f"({qp['fp_count']}/{qp['total_rf_pairs']} real+fake pairs merged)")
    print(f"    False Negative rate: {qp['fn_rate']:.4f}  "
          f"({qp['fn_count']}/{qp['total_rr_pairs']} real+real pairs split)")

    overall_acc = results_df["Correct"].mean()
    print(f"\n  Overall attack accuracy: {overall_acc:.4f}  "
          f"(lower = better privacy for user)")

    print(f"\n  NOTE: Semantic privacy (profile cosine distance) is computed")
    print(f"  in the profile-building phase using zero-shot classification.")
    print(f"  cluster_results.csv contains the cluster assignments needed.")

    # ---- Save outputs ----

    # cluster_results.csv — main output for downstream profile analysis
    results_path = os.path.join(args.output_dir, "cluster_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved cluster assignments  -> {results_path}")

    # similarity_matrix.pkl — N×N matrix for inspection / visualisation
    matrix_path = os.path.join(args.output_dir, "similarity_matrix.pkl")
    with open(matrix_path, "wb") as f:
        pickle.dump({"matrix": sim_matrix, "n_queries": n_queries}, f)
    print(f"  Saved similarity matrix    -> {matrix_path}")

    # attack_metrics.pkl — all numeric metrics for comparison across conditions
    metrics = {
        "n_queries":      n_queries,
        "n_pairs":        n_pairs,
        "real_cluster":   real_cluster,
        "fake_cluster":   fake_cluster,
        "fp_rate":        qp["fp_rate"],
        "fn_rate":        qp["fn_rate"],
        "fp_count":       qp["fp_count"],
        "fn_count":       qp["fn_count"],
        "total_rr_pairs": qp["total_rr_pairs"],
        "total_rf_pairs": qp["total_rf_pairs"],
        "attack_accuracy": overall_acc,
    }
    metrics_path = os.path.join(args.output_dir, "attack_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"  Saved attack metrics       -> {metrics_path}")


if __name__ == "__main__":
    main()
