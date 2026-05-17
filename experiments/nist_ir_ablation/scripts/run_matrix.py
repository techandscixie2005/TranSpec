#!/usr/bin/env python3
"""
Orchestrate the NIST IR ablation 2x2 experiment matrix.

Usage:
  # Full experiment
  python run_matrix.py \\
    --input data/nist_ir/raw/IR_nist.jsonl \\
    --output runs/nist_ir_ablation \\
    --config experiments/nist_ir_ablation/configs/nist_ir_base.yaml \\
    --models E0_atom_nope E1_atom_fourier E2_spe_nope E3_spe_fourier \\
    --seeds 42 2025 3407 \\
    --run_preprocess --run_train --run_eval --aggregate

  # Smoke test (single seed, small model)
  python run_matrix.py \\
    --input data/nist_ir/raw/IR_nist_200.jsonl \\
    --output runs/nist_ir_ablation_smoke \\
    --config experiments/nist_ir_ablation/configs/smoke_200.yaml \\
    --models E0_atom_nope E1_atom_fourier E2_spe_nope E3_spe_fourier \\
    --seeds 42 \\
    --run_preprocess --run_train --run_eval --aggregate

  # Single run (useful for Slurm jobs)
  python run_matrix.py \\
    --output runs/nist_ir_ablation \\
    --config experiments/nist_ir_ablation/configs/nist_ir_base.yaml \\
    --model E0_atom_nope --seed 42 \\
    --run_train --run_eval --skip_preprocess_if_exists

  # Dry-run (print commands, don't execute)
  python run_matrix.py --input ... --output ... --config ... \\
    --run_preprocess --run_train --run_eval --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
import time


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))


def resolve_path(p):
    """Resolve a path relative to project root if not absolute."""
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(_PROJECT_ROOT, p))


def get_python_cmd():
    """Return the Python executable (same as the one running this script)."""
    return sys.executable or "python"


def default_output_dirs(output_root):
    """Derive processed, run, and summary directories from output root."""
    root = resolve_path(output_root)
    return {
        "processed_dir": os.path.join(root, "processed"),
        "run_dir": os.path.join(root, "runs"),
        "summary_dir": os.path.join(root, "summary"),
    }


def run_command(cmd, dry_run=False, capture=True, log_path=None):
    """Run a command or print it in dry-run mode."""
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd

    if dry_run:
        print(f"[DRY-RUN] {cmd_str}")
        return True

    print(f"[RUN] {cmd_str}")
    t0 = time.time()

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as log_f:
            result = subprocess.run(cmd, capture_output=capture, text=True)
            log_f.write(f"STDOUT:\n{result.stdout}\n")
            log_f.write(f"STDERR:\n{result.stderr}\n")
            log_f.write(f"Return code: {result.returncode}\n")
    else:
        result = subprocess.run(cmd, capture_output=capture, text=True)

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s  Return code: {result.returncode}")

    if result.returncode != 0:
        if result.stderr:
            print(f"  STDERR: {result.stderr[:500]}")
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="NIST IR Ablation Experiment Matrix Orchestrator",
    )
    # Input / output
    parser.add_argument("--input", default=None, help="Path to input JSONL file")
    parser.add_argument(
        "--output", default=None,
        help="Output root directory (e.g., runs/nist_ir_ablation)",
    )
    parser.add_argument("--config", required=True, help="Path to base YAML config")
    parser.add_argument(
        "--processed_dir", default=None,
        help="Override processed data directory",
    )
    parser.add_argument(
        "--run_dir", default=None,
        help="Override run output directory",
    )
    parser.add_argument(
        "--summary_dir", default=None,
        help="Override summary output directory",
    )

    # Model selection
    parser.add_argument(
        "--models", nargs="*", default=[],
        help="List of model IDs (e.g., E0_atom_nope E1_atom_fourier)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Single model ID (alternative to --models for Slurm jobs)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=[],
        help="List of random seeds (e.g., 42 2025 3407)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Single seed (alternative to --seeds for Slurm jobs)",
    )

    # Actions
    parser.add_argument(
        "--run_preprocess", action="store_true",
        help="Run preprocessing",
    )
    parser.add_argument(
        "--run_train", action="store_true",
        help="Run training for each model/seed",
    )
    parser.add_argument(
        "--run_eval", action="store_true",
        help="Run evaluation for each model/seed",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Run aggregation after all jobs complete",
    )
    parser.add_argument(
        "--skip_preprocess_if_exists", action="store_true",
        help="Skip preprocessing if processed data already exists",
    )

    # Decoding / runtime config
    parser.add_argument(
        "--decode_method", default="beam",
        choices=["beam", "threshold"],
        help="Decoding method",
    )
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--candidate_limit", type=int, default=20)
    parser.add_argument("--max_decode_len", type=int, default=256)
    parser.add_argument(
        "--continue_on_error", action="store_true",
        help="Continue running remaining jobs if one fails",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--base_config", default=None,
        help="Override base config path for train/eval scripts",
    )

    return parser.parse_args()


def find_condition_config(model_id):
    """Find the condition-specific YAML config for a model ID."""
    config_dir = os.path.join(_SCRIPT_DIR, "..", "configs")
    for ext in ["yaml", "yml"]:
        path = os.path.join(config_dir, f"{model_id}.{ext}")
        if os.path.exists(path):
            return path
    return None


def get_base_config(args):
    """Determine the base config to pass to train/eval scripts."""
    if args.base_config:
        return resolve_path(args.base_config)
    # Default to the same config used for preprocessing
    return resolve_path(args.config)


def main():
    args = parse_args()

    # Resolve paths
    config_path = resolve_path(args.config)

    # Determine output directories
    if args.output:
        derived = default_output_dirs(args.output)
        processed_dir = args.processed_dir or derived["processed_dir"]
        run_dir = args.run_dir or derived["run_dir"]
        summary_dir = args.summary_dir or derived["summary_dir"]
    else:
        processed_dir = resolve_path(args.processed_dir) if args.processed_dir else None
        run_dir = resolve_path(args.run_dir) if args.run_dir else None
        summary_dir = resolve_path(args.summary_dir) if args.summary_dir else None

    if not processed_dir:
        raise SystemExit("ERROR: --processed_dir or --output is required")

    python_cmd = get_python_cmd()

    # Resolve model IDs
    model_ids = args.models[:]
    if args.model and args.model not in model_ids:
        model_ids.append(args.model)
    seed_list = args.seeds[:]
    if args.seed is not None and args.seed not in seed_list:
        seed_list.append(args.seed)

    if model_ids == [] or seed_list == []:
        # For single-run mode, construct from args
        if args.model and args.seed is not None:
            model_ids = [args.model]
            seed_list = [args.seed]
        else:
            raise SystemExit(
                "ERROR: Specify --models/--model and --seeds/--seed"
            )

    # Print experiment summary
    print("=" * 60)
    print("  NIST IR Ablation Experiment Matrix")
    print("=" * 60)
    print(f"  Config: {config_path}")
    print(f"  Models: {model_ids}")
    print(f"  Seeds: {seed_list}")
    print(f"  Total runs: {len(model_ids) * len(seed_list)}")
    if args.output:
        print(f"  Output: {resolve_path(args.output)}")
    print(f"  Processed dir: {processed_dir}")
    print(f"  Run dir: {run_dir}")
    print(f"  Summary dir: {summary_dir}")
    print(f"  Decode method: {args.decode_method}")
    if args.decode_method == "beam":
        print(f"  Beam size: {args.beam_size}, Candidate limit: {args.candidate_limit}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 60)

    # =========================================================
    # Preprocess
    # =========================================================
    if args.run_preprocess:
        print("\n" + "=" * 60)
        print("  PREPROCESSING")
        print("=" * 60)

        if args.skip_preprocess_if_exists:
            manifest_path = os.path.join(processed_dir, "common", "preprocess_manifest.json")
            if os.path.exists(manifest_path):
                print(f"  Existing processed data found at {processed_dir}, skipping.")
            else:
                args.skip_preprocess_if_exists = False  # Don't skip, no data exists

        if not args.skip_preprocess_if_exists or not os.path.exists(
            os.path.join(processed_dir, "common", "preprocess_manifest.json")
        ):
            if not args.input:
                raise SystemExit("ERROR: --input is required for preprocessing")

            prep_cmd = [
                python_cmd,
                os.path.join(_SCRIPT_DIR, "prepare_jsonl.py"),
                "--input", resolve_path(args.input),
                "--output", resolve_path(args.output) if args.output else processed_dir,
                "--config", config_path,
            ]
            if args.dry_run:
                print(f"  [DRY-RUN] {' '.join(prep_cmd)}")
            else:
                success = run_command(prep_cmd, dry_run=False, log_path=os.path.join(processed_dir, "common", "preprocess.log"))
                if not success:
                    raise SystemExit("ERROR: Preprocessing failed")
        else:
            print(f"  Processed data already exists at {processed_dir}, skipping.")
    else:
        print(f"\n  Skipping preprocessing (use --run_preprocess to enable).")

    # =========================================================
    # Train + Evaluate
    # =========================================================
    base_config = get_base_config(args)
    total_runs = len(model_ids) * len(seed_list)
    completed = 0
    failed = 0

    for model_id in model_ids:
        # Find condition config
        cond_config = find_condition_config(model_id)
        if cond_config is None:
            print(f"  WARNING: Condition config not found for {model_id}, skipping.")
            failed += len(seed_list)
            continue

        for seed_val in seed_list:
            run_label = f"{model_id}/seed_{seed_val}"
            print(f"\n{'=' * 60}")
            print(f"  RUN: {run_label}")
            print(f"{'=' * 60}")

            run_out_dir = os.path.join(run_dir, model_id, f"seed_{seed_val}")

            # === TRAIN ===
            if args.run_train:
                ckpt_path = os.path.join(run_out_dir, "checkpoints", "best.pt")
                if os.path.exists(ckpt_path):
                    print(f"  Checkpoint exists at {ckpt_path}, skipping training.")
                else:
                    train_cmd = [
                        python_cmd,
                        os.path.join(_SCRIPT_DIR, "train_one.py"),
                        "--config", cond_config,
                        "--processed_dir", processed_dir,
                        "--output", run_out_dir,
                        "--seed", str(seed_val),
                        "--base_config", base_config,
                    ]
                    log_path = os.path.join(run_out_dir, "logs", "train_stdout.log")
                    success = run_command(train_cmd, dry_run=args.dry_run, log_path=log_path)
                    if not success:
                        msg = f"ERROR: Training failed for {run_label}"
                        if args.continue_on_error:
                            print(f"  {msg}")
                            failed += 1
                            continue
                        else:
                            raise SystemExit(msg)
            else:
                print(f"  Skipping training for {run_label}.")

            # === EVALUATE ===
            if args.run_eval:
                eval_out = os.path.join(run_out_dir, "eval")
                metrics_path = os.path.join(eval_out, "metrics.json")
                if os.path.exists(metrics_path):
                    print(f"  Evaluation exists at {metrics_path}, skipping.")
                else:
                    # Use best.pt if available, fall back to last.pt
                    checkpoint = os.path.join(run_out_dir, "checkpoints", "best.pt")
                    if not os.path.exists(checkpoint):
                        checkpoint = os.path.join(run_out_dir, "checkpoints", "last.pt")
                    if not os.path.exists(checkpoint):
                        print(f"  WARNING: No checkpoint found for {run_label}, skipping eval.")
                        continue

                    eval_cmd = [
                        python_cmd,
                        os.path.join(_SCRIPT_DIR, "evaluate_one.py"),
                        "--config", cond_config,
                        "--processed_dir", processed_dir,
                        "--checkpoint", checkpoint,
                        "--output", eval_out,
                        "--seed", str(seed_val),
                        "--base_config", base_config,
                        "--decode_method", args.decode_method,
                        "--beam_size", str(args.beam_size),
                        "--candidate_limit", str(args.candidate_limit),
                        "--max_decode_len", str(args.max_decode_len),
                    ]
                    eval_log = os.path.join(run_out_dir, "logs", "eval_stdout.log")
                    success = run_command(eval_cmd, dry_run=args.dry_run, log_path=eval_log)
                    if not success:
                        msg = f"ERROR: Evaluation failed for {run_label}"
                        if args.continue_on_error:
                            print(f"  {msg}")
                            failed += 1
                            continue
                        else:
                            raise SystemExit(msg)
            else:
                print(f"  Skipping evaluation for {run_label}.")

            completed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Runs completed: {completed}, Failed: {failed}")
    print(f"{'=' * 60}")

    # =========================================================
    # Aggregate
    # =========================================================
    if args.aggregate and run_dir and summary_dir:
        print("\n" + "=" * 60)
        print("  AGGREGATING RESULTS")
        print("=" * 60)

        agg_cmd = [
            python_cmd,
            os.path.join(_SCRIPT_DIR, "aggregate_results.py"),
            "--run_dir", run_dir,
            "--summary_dir", summary_dir,
        ]
        success = run_command(agg_cmd, dry_run=args.dry_run)
        if not success:
            print("  WARNING: Aggregation encountered issues (partial results).")

        # Report
        report_cmd = [
            python_cmd,
            os.path.join(_SCRIPT_DIR, "make_report.py"),
            "--summary_dir", summary_dir,
            "--output_md", os.path.join(summary_dir, "report.md"),
            "--output_html", os.path.join(summary_dir, "report.html"),
        ]
        success = run_command(report_cmd, dry_run=args.dry_run)
        if not success:
            print("  WARNING: Report generation encountered issues.")

        if not args.dry_run:
            report_path = os.path.join(summary_dir, "report.md")
            if os.path.exists(report_path):
                print(f"\n  Report: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
