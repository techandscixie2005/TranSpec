#!/usr/bin/env python3
"""
Aggregate NIST IR ablation experiment results.

Reads metrics.json files from run directories, computes summary statistics,
ablation effects, and writes CSVs and JSON outputs.

Usage:
  python aggregate_results.py \\
    --run_dir runs/nist_ir_ablation/runs \\
    --summary_dir runs/nist_ir_ablation/summary

Optional:
  --strict        Require all 12 runs (exit 1 if any missing)
  --merge_spe     Treat atom-only models as SPE=False
                  (default: auto-detect from model_id pattern)
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate NIST IR ablation results",
    )
    parser.add_argument(
        "--run_dir", required=True,
        help="Run output directory (e.g., runs/nist_ir_ablation/runs)",
    )
    parser.add_argument(
        "--summary_dir", required=True,
        help="Summary output directory",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Require all 12 expected runs",
    )
    return parser.parse_args()


# Expected experiment matrix
EXPECTED_MODELS = ["E0_atom_nope", "E1_atom_fourier", "E2_spe_nope", "E3_spe_fourier"]
EXPECTED_SEEDS = [42, 2025, 3407]


def discover_runs(run_dir):
    """Discover completed runs by scanning directory structure.

    Looks for runs/{model_id}/seed_{seed}/eval/metrics.json.

    Returns:
        runs: dict keyed by (model_id, seed) with metrics data
    """
    runs = {}
    missing = []

    if not os.path.isdir(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}")
        return {}, []

    for model_id in EXPECTED_MODELS:
        model_path = os.path.join(run_dir, model_id)
        if not os.path.isdir(model_path):
            missing.append({"model": model_id, "seeds": EXPECTED_SEEDS[:]})
            continue

        for seed in EXPECTED_SEEDS:
            seed_path = os.path.join(model_path, f"seed_{seed}")
            metrics_path = os.path.join(seed_path, "eval", "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                runs[(model_id, seed)] = metrics
            else:
                missing.append({"model": model_id, "seed": seed})

    return runs, missing


def compute_mean_std(values):
    """Compute mean and sample std (ddof=1)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, variance ** 0.5


def aggregate_by_model(runs):
    """Group runs by model_id and compute mean/std for each metric."""
    from collections import defaultdict

    model_groups = defaultdict(list)
    for (model_id, seed), metrics in runs.items():
        model_groups[model_id].append(metrics)

    summary = {}
    for model_id in EXPECTED_MODELS:
        group = model_groups.get(model_id, [])
        if not group:
            summary[model_id] = {"runs": 0}
            continue

        metrics_of_interest = [
            "top1", "top3", "top5", "top10",
            "invalid_decode_rate", "avg_num_candidates",
            "total_params", "trainable_params",
            "train_time_sec", "eval_time_sec",
        ]

        entry = {"runs": len(group)}
        for key in metrics_of_interest:
            values = [m.get(key, 0.0) or 0.0 for m in group]
            mean_val, std_val = compute_mean_std(values)
            entry[f"{key}_mean"] = mean_val
            entry[f"{key}_std"] = std_val

        summary[model_id] = entry

    return summary


def write_all_runs_csv(runs, path):
    """Write all_runs.csv with one row per completed run."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "model_id", "seed",
        "top1", "top3", "top5", "top10",
        "invalid_decode_count", "invalid_decode_rate",
        "avg_num_candidates",
        "decode_method", "beam_size", "candidate_limit",
        "total_params", "trainable_params",
        "num_test", "best_epoch", "best_valid_loss",
        "train_time_sec", "eval_time_sec",
    ]

    rows = []
    for (model_id, seed), metrics in sorted(runs.items()):
        row = {"model_id": model_id, "seed": seed}
        for key in fieldnames:
            if key in ("model_id", "seed"):
                continue
            row[key] = metrics.get(key, "")
        rows.append(row)

    with open(path, "w") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} rows to {path}")


def write_summary_csv(summary, path):
    """Write summary_mean_std.csv."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    metrics = ["top1", "top3", "top5", "top10",
               "invalid_decode_rate", "avg_num_candidates",
               "total_params", "train_time_sec", "eval_time_sec"]

    fieldnames = ["model_id"] + [f"{m}_{stat}" for m in metrics for stat in ("mean", "std")]

    rows = []
    for model_id in EXPECTED_MODELS:
        entry = summary.get(model_id, {"runs": 0})
        if entry["runs"] == 0:
            row = {"model_id": model_id}
            for m in metrics:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
            rows.append(row)
            continue
        row = {"model_id": model_id}
        for m in metrics:
            row[f"{m}_mean"] = entry.get(f"{m}_mean", "")
            row[f"{m}_std"] = entry.get(f"{m}_std", "")
        rows.append(row)

    with open(path, "w") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} rows to {path}")


def compute_ablation_effects(summary):
    """Compute ablation effects from mean values.

    E0: atom_nope     (baseline: atom tokenizer, no Fourier)
    E1: atom_fourier  (Fourier PE only)
    E2: spe_nope      (SPE tokenizer only)
    E3: spe_fourier   (both Fourier + SPE)

    delta_pe      = E1 - E0  (effect of Fourier PE with atom tokenizer)
    delta_spe     = E2 - E0  (effect of SPE with no Fourier)
    delta_both    = E3 - E0  (effect of both)
    interaction   = (E3 - E2) - (E1 - E0)
    """
    metrics = ["top1", "top3", "top5", "top10"]

    def get_mean(model_id, metric):
        entry = summary.get(model_id, {})
        if entry.get("runs", 0) == 0:
            return None
        return entry.get(f"{metric}_mean", None)

    effects = {}
    for metric in metrics:
        e0 = get_mean("E0_atom_nope", metric)
        e1 = get_mean("E1_atom_fourier", metric)
        e2 = get_mean("E2_spe_nope", metric)
        e3 = get_mean("E3_spe_fourier", metric)

        if any(v is None for v in [e0, e1, e2, e3]):
            effects[metric] = {
                "delta_pe": None,
                "delta_spe": None,
                "delta_both": None,
                "interaction": None,
                "note": "some conditions missing",
            }
        else:
            effects[metric] = {
                "delta_pe": round(e1 - e0, 6),
                "delta_spe": round(e2 - e0, 6),
                "delta_both": round(e3 - e0, 6),
                "interaction": round((e3 - e2) - (e1 - e0), 6),
            }

    return effects


def main():
    args = parse_args()

    run_dir = os.path.abspath(args.run_dir)
    summary_dir = os.path.abspath(args.summary_dir)

    print(f"Run dir: {run_dir}")
    print(f"Summary dir: {summary_dir}")

    # Discover runs
    runs, missing = discover_runs(run_dir)
    print(f"\nDiscovered {len(runs)} completed run(s)")
    if missing:
        print(f"  Missing: {len(missing)} run(s)")
        for m in missing:
            model = m.get("model", "")
            seed = m.get("seed", "")
            if seed:
                print(f"    {model}/seed_{seed}")
            else:
                print(f"    {model} (no runs found)")

    # Strict mode
    expected_total = len(EXPECTED_MODELS) * len(EXPECTED_SEEDS)
    if args.strict and len(runs) < expected_total:
        print(
            f"\nERROR: --strict mode: expected {expected_total} runs, "
            f"found {len(runs)}"
        )
        sys.exit(1)

    if len(runs) == 0:
        print("\nERROR: No completed runs found.")
        sys.exit(1)

    # Aggregate by model
    summary = aggregate_by_model(runs)

    # Write CSVs
    all_runs_path = os.path.join(summary_dir, "all_runs.csv")
    write_all_runs_csv(runs, all_runs_path)

    summary_csv_path = os.path.join(summary_dir, "summary_mean_std.csv")
    write_summary_csv(summary, summary_csv_path)

    # Ablation effects
    effects = compute_ablation_effects(summary)
    effects_path = os.path.join(summary_dir, "ablation_effects.json")
    with open(effects_path, "w") as f:
        json.dump(effects, f, indent=2)
    print(f"  Wrote ablation effects to {effects_path}")

    # Missing runs
    if missing:
        missing_path = os.path.join(summary_dir, "missing_runs.json")
        # Deduplicate missing
        missing_dedup = {}
        for m in missing:
            key = m.get("model", "") + "_" + str(m.get("seed", ""))
            if key not in missing_dedup:
                missing_dedup[key] = m
        with open(missing_path, "w") as f:
            json.dump({"missing": list(missing_dedup.values())}, f, indent=2)
        print(f"  Wrote missing runs to {missing_path}")
    else:
        missing_path = os.path.join(summary_dir, "missing_runs.json")
        with open(missing_path, "w") as f:
            json.dump({"missing": []}, f, indent=2)
        print(f"  No missing runs.")

    # Print summary table
    print(f"\n{'=' * 60}")
    print("  Model                    Top-1    Top-3    Top-5    Top-10")
    print(f"{'=' * 60}")
    for model_id in EXPECTED_MODELS:
        entry = summary.get(model_id, {"runs": 0})
        if entry["runs"] == 0:
            print(f"  {model_id:<24}  --       --       --       --")
            continue
        def fmt_mean_std(key):
            m = entry.get(f"{key}_mean", 0)
            s = entry.get(f"{key}_std", 0)
            return f"{m:.4f}±{s:.4f}"
        t1 = fmt_mean_std("top1")
        t3 = fmt_mean_std("top3")
        t5 = fmt_mean_std("top5")
        t10 = fmt_mean_std("top10")
        print(f"  {model_id:<24}  {t1}  {t3}  {t5}  {t10}")

    # Print ablation effects
    print(f"\n{'=' * 60}")
    print("  Ablation Effects (difference in Top-k accuracy)")
    print(f"{'=' * 60}")
    header = f"  {'Effect':<20}"
    for metric in ["top1", "top3", "top5", "top10"]:
        header += f"  {metric:>8}"
    print(header)
    print(f"  {'-' * 20}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for effect_name in ["delta_pe", "delta_spe", "delta_both", "interaction"]:
        row = f"  {effect_name:<20}"
        for metric in ["top1", "top3", "top5", "top10"]:
            val = effects.get(metric, {}).get(effect_name, None)
            if val is not None:
                row += f"  {val:>+8.4f}"
            else:
                row += f"  {'N/A':>8}"
        print(row)

    print(f"\nSummary saved to {summary_dir}")


if __name__ == "__main__":
    main()
