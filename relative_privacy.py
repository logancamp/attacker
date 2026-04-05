
import argparse
import pickle
import os
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Relative Privacy Evaluation")
    p.add_argument("--control", required=True, help="Path to control output dir")
    p.add_argument("--experiment", required=True, help="Path to experiment output dir")
    p.add_argument("--output_dir", default="output", help="Where to save results")
    return p.parse_args()


def load_metrics(output_dir):
    path = os.path.join(output_dir, "attack_metrics.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    control = load_metrics(args.control)
    experiment = load_metrics(args.experiment)

    metrics_to_compare = [
        "silhouette_score",
        "attack_accuracy",
        "fp_rate",
        "fn_rate",
    ]

    # build the comparison table
    rows = []
    for metric in metrics_to_compare:
        c_val = control[metric]
        e_val = experiment[metric]
        delta = e_val - c_val
        pct_change = (delta / c_val * 100) if c_val != 0 else float('inf')

        rows.append({
            "Metric": metric,
            "Control": round(c_val, 4),
            "Experiment": round(e_val, 4),
            "Delta": round(delta, 4),
            "Pct_Change": round(pct_change, 2),
        })

    # print it
    print(f"\n{'='*70}")
    print(f"  RELATIVE PRIVACY EVALUATION")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<20s} {'Control':>10s} {'Experiment':>10s} {'Delta':>10s} {'% Change':>10s}")
    print(f"  {'-'*60}")
    for r in rows:
        print(f"  {r['Metric']:<20s} {r['Control']:>10.4f} {r['Experiment']:>10.4f} {r['Delta']:>+10.4f} {r['Pct_Change']:>+9.2f}%")

    # save it
    df = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, "relative_privacy.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
