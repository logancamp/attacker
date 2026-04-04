"""
PHASE 4: GBRT TRAINING
Train N_MODELS GBRT classifiers on different random subsets of the training pairs.

Why 60 models instead of one?
  The adversary does not know which subset of training users will best generalise
  to the target user. Training on different subsets and taking the median score
  (phase 5) cancels out subset-specific bias without requiring knowledge of the
  target user. See paper Section 4.1.3.

Input : output_dir/train_pairs.pkl   (from phase 3, train mode)
Output: output_dir/models/model_XX.pkl  (one file per model)
        output_dir/models/metadata.pkl  (feature importance summary)
"""

import argparse
import os
import pickle
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: GBRT Training")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--n_models", type=int, default=60)
    parser.add_argument("--subset_frac", type=float, default=0.7)
    return parser.parse_args()


# Single model training
def train_one_model(X, y, subset_frac, random_state):
    """
    Train one GBRT on a random subset of the training data.

    GBRT hyperparameters:
      n_estimators  - number of boosting rounds (trees). 100 is standard.
      max_depth     - shallow trees (3) are weak learners, required for boosting.
      learning_rate - shrinkage; lower = more trees needed but better generalisation.
      subsample     - stochastic gradient boosting: use 80% of data per tree.
                      Adds randomness, improves robustness.

    Each call uses a different random_state, producing a different
    subset selection AND different internal randomness in the GBRT itself.
    """
    n_samples = max(2, int(len(X) * subset_frac))
    rng = np.random.RandomState(random_state)

    # Sample without replacement — each model trained on a different subset
    idx = rng.choice(len(X), size=n_samples, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    # Guard: ensure both classes present in subset
    if len(np.unique(y_sub)) < 2:
        # If only one class sampled (very small dataset), fall back to full data
        X_sub, y_sub = X, y

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=random_state,
    )
    model.fit(X_sub, y_sub)
    return model


# Entry point
def main():
    args = parse_args()
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n[PHASE 4] GBRT Training")
    print(f"  Pairs  : {args.pairs}")
    print(f"  Models : {models_dir}")
    print(f"  N      : {args.n_models} models  |  subset_frac={args.subset_frac}\n")

    # Load training data
    with open(args.pairs, "rb") as f:
        data = pickle.load(f)

    X = data["X"] # shape (N_pairs, 27)
    y = data["y"] # shape (N_pairs,)  values 0/1
    feature_names = data["feature_names"] # list of 27 names

    print(f"  Training data: {len(X):,} pairs | {X.shape[1]} features")
    print(f"  Class balance: {int(y.sum())} positive (same-source) | "
          f"{int((y == 0).sum())} negative (real+fake)\n")

    # Train N models, each on a different random subset
    importances = []

    for i in tqdm(range(args.n_models), desc="  Training models"):
        # Different random_state per model → different subset + different GBRT seed
        model = train_one_model(X, y, args.subset_frac, random_state=i * 17 + 3)

        model_path = os.path.join(models_dir, f"model_{i:02d}.pkl")
        joblib.dump(model, model_path)

        importances.append(model.feature_importances_)

    # Feature importance summary
    mean_imp = np.mean(importances, axis=0)
    # Normalise to 0-100 range (matches Table 4/5 presentation in paper)
    if mean_imp.max() > 0:
        mean_imp_norm = 100.0 * mean_imp / mean_imp.max()
    else:
        mean_imp_norm = mean_imp

    importance_dict = dict(zip(feature_names, mean_imp_norm))
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    print("\n  Feature importance (normalised 0-100, averaged across models):")
    for feat, imp in sorted_imp:
        bar = "█" * int(imp / 5)
        print(f"    {feat:<28s} {imp:5.1f}  {bar}")

    # Save metadata so phase 5 and downstream analysis can reference it
    meta = {
        "n_models":               args.n_models,
        "subset_frac":            args.subset_frac,
        "feature_names":          feature_names,
        "mean_importance_normed": importance_dict,
        "sorted_importance":      sorted_imp,
    }
    meta_path = os.path.join(models_dir, "metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n  Saved {args.n_models} models + metadata -> {models_dir}/")


if __name__ == "__main__":
    main()
