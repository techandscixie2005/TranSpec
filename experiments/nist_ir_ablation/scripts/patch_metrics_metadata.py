#!/usr/bin/env python3
"""Post-process metrics.json files to add training metadata from checkpoints.

The current Slurm run (job 984501) was started before evaluate_one.py was
updated to extract best_epoch, best_valid_loss, total_params, and
trainable_params from the model. This script patches the existing
metrics.json files so the aggregation produces complete results.

Usage:
  python patch_metrics_metadata.py --run_dir runs/nist_ir_ablation/runs
"""

import argparse
import json
import os
import sys

import torch

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for p in [_PROJECT_ROOT, _SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.nist_ir_ablation.transpec_ext.model_factory import (
    build_model,
    count_parameters,
)
from experiments.nist_ir_ablation.transpec_ext.dataset import load_split_dataset


def count_params_from_checkpoint(checkpoint_path, config, vocab_size, device):
    """Rebuild model and count params."""
    model = build_model(config, vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return count_parameters(model)


def patch_metrics(run_dir, dry_run=False):
    """Patch a single run's metrics.json with checkpoint metadata."""
    metrics_path = os.path.join(run_dir, "eval", "metrics.json")
    checkpoint_path = os.path.join(run_dir, "checkpoints", "best.pt")
    config_path = os.path.join(run_dir, "config.resolved.yaml")

    if not os.path.exists(metrics_path):
        print(f"  SKIP (no metrics.json): {run_dir}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"  SKIP (no best.pt): {run_dir}")
        return
    if not os.path.exists(config_path):
        print(f"  SKIP (no config.resolved.yaml): {run_dir}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Check if already patched
    if metrics.get("total_params", 0) > 0 and metrics.get("best_epoch", 0) > 0:
        print(f"  OK (already has metadata): {run_dir}")
        return

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tokenizer_type = config.get("tokenizer", {}).get("type", "atom")
    vocab_size = metrics.get("vocab_size", 0)

    if vocab_size == 0:
        # Try to determine vocab size from tokenizer
        processed_dir = os.path.dirname(os.path.dirname(run_dir))
        processed_dir = os.path.join(os.path.dirname(processed_dir), "processed")
        if tokenizer_type == "spe":
            from experiments.nist_ir_ablation.transpec_ext.tokenizer_spe import SPETokenizer
            spe_stem = os.path.join(processed_dir, "spe", "spe")
            tokenizer = SPETokenizer.load(spe_stem)
        else:
            from experiments.nist_ir_ablation.transpec_ext.tokenizer_atom import AtomTokenizer
            vocab_path = os.path.join(processed_dir, "atom", "atom_vocab.json")
            tokenizer = AtomTokenizer.load(vocab_path)
        vocab_size = tokenizer.vocab_size

    # Count params
    param_counts = count_params_from_checkpoint(
        checkpoint_path, config, vocab_size, device="cpu",
    )

    # Extract checkpoint metadata
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_epoch = checkpoint.get("epoch", 0)
    best_valid_loss = checkpoint.get("valid_loss", 0.0)

    # Update metrics
    metrics["total_params"] = param_counts["total_params"]
    metrics["trainable_params"] = param_counts["trainable_params"]
    metrics["best_epoch"] = best_epoch
    metrics["best_valid_loss"] = best_valid_loss

    if dry_run:
        print(f"  WOULD UPDATE: {run_dir}")
        print(f"    total_params={param_counts['total_params']}, "
              f"trainable_params={param_counts['trainable_params']}")
        print(f"    best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.6f}")
    else:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  PATCHED: {run_dir}")
        print(f"    total_params={param_counts['total_params']}, "
              f"trainable_params={param_counts['trainable_params']}")
        print(f"    best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Patch metrics.json files with training metadata from checkpoints",
    )
    parser.add_argument(
        "--run_dir", required=True,
        help="Run output directory (e.g., runs/nist_ir_ablation/runs)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Preview changes without writing",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Patching metrics.json with training metadata")
    print("=" * 60)
    print(f"  Run dir: {run_dir}")
    if args.dry_run:
        print("  Mode: DRY RUN (no changes written)")
    print()

    patched = 0
    skipped = 0
    for model_id in sorted(os.listdir(run_dir)):
        model_path = os.path.join(run_dir, model_id)
        if not os.path.isdir(model_path):
            continue
        for seed_dir in sorted(os.listdir(model_path)):
            if not seed_dir.startswith("seed_"):
                continue
            seed_path = os.path.join(model_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue
            patch_metrics(seed_path, dry_run=args.dry_run)
            patched += 1

    print()
    print(f"  Total attempted: {patched}")


if __name__ == "__main__":
    main()
