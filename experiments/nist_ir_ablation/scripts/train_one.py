#!/usr/bin/env python3
"""
Train a single ablation model for NIST IR experiments.

Trains E0_atom_nope or E1_atom_fourier with teacher forcing on preprocessed data.
Saves checkpoints, logs, and config snapshot.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add project root and src/ for legacy imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for p in [_PROJECT_ROOT, _SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.nist_ir_ablation.transpec_ext.dataset import (
    ablation_collate_fn,
    load_split_dataset,
)
from experiments.nist_ir_ablation.transpec_ext.io_utils import save_json, save_torch
from experiments.nist_ir_ablation.transpec_ext.model_factory import (
    build_model,
    count_parameters,
)
from experiments.nist_ir_ablation.transpec_ext.tokenizer_atom import AtomTokenizer
from experiments.nist_ir_ablation.transpec_ext.vocab import PAD_ID
from torch.utils.data import DataLoader


def deep_update(base, overrides):
    """Recursively update a nested dict with overrides."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a single NIST IR ablation model"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to condition-specific YAML (e.g., E0_atom_nope.yaml)",
    )
    parser.add_argument(
        "--processed_dir",
        required=True,
        help="Path to processed data root (e.g., runs/.../processed)",
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for checkpoints and logs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--base_config", default=None,
        help="Base YAML config path (default: nist_ir_base.yaml, fallback smoke_200.yaml)",
    )
    return parser.parse_args()


def load_configs(args):
    """Load and merge condition config over base config."""
    base_path = args.base_config
    if base_path is None:
        config_dir = os.path.dirname(os.path.abspath(args.config))
        base_path = os.path.join(config_dir, "smoke_200.yaml")
    if not os.path.exists(base_path):
        print(f"ERROR: Base config not found at {base_path}")
        sys.exit(1)

    with open(base_path) as f:
        config = yaml.safe_load(f)
    with open(args.config) as f:
        cond_cfg = yaml.safe_load(f)

    config = deep_update(config, cond_cfg)

    # CLI overrides
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    config["seed"] = args.seed

    return config


def train_one_epoch(
    model, loader, optimizer, criterion, device, grad_clip, epoch_info
):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        spectrum = batch["spectrum"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        labels = batch["labels"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)
        tgt_padding_mask = batch["tgt_padding_mask"].to(device)

        optimizer.zero_grad()
        logits = model(spectrum, decoder_input, tgt_mask, tgt_padding_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation. Returns average loss."""
    model.eval()
    total_loss = 0.0
    for batch in loader:
        spectrum = batch["spectrum"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        labels = batch["labels"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)
        tgt_padding_mask = batch["tgt_padding_mask"].to(device)

        logits = model(spectrum, decoder_input, tgt_mask, tgt_padding_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item()

    return total_loss / len(loader)


def save_csv(path, rows):
    """Save list of dicts as CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    config = load_configs(args)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = os.path.splitext(os.path.basename(args.config))[0]
    use_fourier = config.get("spectral_fourier_encoding", False)
    tokenizer_type = config.get("tokenizer", {}).get("type", "atom")
    training_cfg = config.get("training", {})

    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Fourier: {use_fourier}")
    print(f"Tokenizer: {tokenizer_type}")
    print(f"Output: {args.output}")

    # Load tokenizer to determine vocab_size
    if tokenizer_type == "spe":
        from experiments.nist_ir_ablation.transpec_ext.tokenizer_spe import (
            SPETokenizer,
        )

        spe_stem = os.path.join(args.processed_dir, "spe", "spe")
        tokenizer = SPETokenizer.load(spe_stem)
    else:
        tokenizer = AtomTokenizer.load(
            os.path.join(args.processed_dir, "atom", "atom_vocab.json")
        )

    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    ckpt_dir = os.path.join(args.output, "checkpoints")
    log_dir = os.path.join(args.output, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save resolved config
    resolved_config = {
        "model_id": model_id,
        "tokenizer_type": tokenizer_type,
        "fourier_enabled": use_fourier,
        "vocab_size": vocab_size,
        "seed": args.seed,
        "config": config,
    }
    save_json(resolved_config, os.path.join(args.output, "config.resolved.json"))
    with open(os.path.join(args.output, "config.resolved.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Determine target dir for tokenized data
    target_dir = "spe" if tokenizer_type == "spe" else "atom"

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_split_dataset(args.processed_dir, "train", target_dir=target_dir)
    valid_dataset = load_split_dataset(args.processed_dir, "valid", target_dir=target_dir)
    print(f"  Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get("batch_size", 32),
        shuffle=True,
        collate_fn=ablation_collate_fn,
        num_workers=training_cfg.get("num_workers", 0),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_cfg.get("batch_size", 32),
        shuffle=False,
        collate_fn=ablation_collate_fn,
        num_workers=training_cfg.get("num_workers", 0),
    )

    # Build model
    print("Building model...")
    model = build_model(config, vocab_size)
    model = model.to(device)
    param_counts = count_parameters(model)
    print(f"  Total params: {param_counts['total_params']:,}")
    print(f"  Trainable params: {param_counts['trainable_params']:,}")

    # Save model summary
    summary_lines = [
        "=" * 60,
        "Model Summary",
        "=" * 60,
        f"Model ID:       {model_id}",
        f"Tokenizer type: {tokenizer_type}",
        f"Fourier:        {'enabled' if use_fourier else 'disabled'}",
        f"Vocab size:     {vocab_size}",
        "",
        "Parameters:",
        f"  Total:     {param_counts['total_params']:,}",
        f"  Trainable: {param_counts['trainable_params']:,}",
        "",
        "Hyperparameters:",
        f"  d_model:              {config.get('model', {}).get('d_model', 'N/A')}",
        f"  nhead:                {config.get('model', {}).get('nhead', 'N/A')}",
        f"  num_encoder_layers:   {config.get('model', {}).get('num_encoder_layers', 'N/A')}",
        f"  num_decoder_layers:   {config.get('model', {}).get('num_decoder_layers', 'N/A')}",
        f"  dim_feedforward:     {config.get('model', {}).get('dim_feedforward', 'N/A')}",
        f"  dropout:             {config.get('model', {}).get('dropout', 'N/A')}",
        f"  batch_size:           {training_cfg.get('batch_size', 'N/A')}",
        f"  learning_rate:        {training_cfg.get('lr', 'N/A')}",
        f"  weight_decay:         {training_cfg.get('weight_decay', 'N/A')}",
        f"  grad_clip:            {training_cfg.get('grad_clip', 'N/A')}",
        f"  label_smoothing:      {training_cfg.get('label_smoothing', 'N/A')}",
        f"  max_len (decoding):   {config.get('decoding', {}).get('max_len', 'N/A')}",
        f"  fourier_num_freqs:    {config.get('fourier', {}).get('num_frequencies', 'N/A')}",
        "",
        "Legacy status:",
        "  src/model.py:  not modified",
        "  src/util.py:   not modified",
        "=" * 60,
    ]
    summary_path = os.path.join(args.output, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  Summary saved to {summary_path}")

    # Optimizer, loss, LR scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("lr", 2e-4),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        betas=tuple(training_cfg.get("betas", (0.9, 0.999))),
    )

    label_smoothing = training_cfg.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)
    grad_clip = training_cfg.get("grad_clip", 1.0)

    scheduler = None
    scheduler_cfg = config.get("scheduler", {})
    if scheduler_cfg.get("type") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 5),
        )

    # Training loop
    num_epochs = training_cfg.get("epochs", 10)
    best_valid_loss = float("inf")
    patience_counter = 0
    patience = training_cfg.get("early_stopping_patience", 0)
    train_log = []
    valid_log = []

    print(f"\nTraining for {num_epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip, epoch
        )
        train_elapsed = time.time() - t0

        val_loss = validate(model, valid_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]

        # Log
        train_log.append(
            {
                "epoch": epoch,
                "loss": round(train_loss, 6),
                "learning_rate": lr,
                "elapsed_time_sec": round(train_elapsed, 2),
            }
        )
        valid_log.append(
            {
                "epoch": epoch,
                "loss": round(val_loss, 6),
                "learning_rate": lr,
                "elapsed_time_sec": round(train_elapsed, 2),
            }
        )

        print(
            f"  Epoch {epoch:>2}/{num_epochs}: "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={val_loss:.4f} "
            f"lr={lr:.2e} "
            f"({train_elapsed:.1f}s)"
        )

        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save best checkpoint
        is_best = val_loss < best_valid_loss
        if is_best:
            best_valid_loss = val_loss
            patience_counter = 0
            save_torch(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": val_loss,
                },
                os.path.join(ckpt_dir, "best.pt"),
            )
        else:
            patience_counter += 1

        # Save last checkpoint
        save_torch(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": val_loss,
            },
            os.path.join(ckpt_dir, "last.pt"),
        )

        # Early stopping
        if patience > 0 and patience_counter >= patience:
            print(f"  Early stopping triggered (patience={patience})")
            break

    # Save logs
    save_csv(os.path.join(log_dir, "train_log.csv"), train_log)
    save_csv(os.path.join(log_dir, "valid_log.csv"), valid_log)
    print(f"\nLogs saved to {log_dir}")
    print(f"Checkpoints saved to {ckpt_dir}")
    print("Training complete.")


if __name__ == "__main__":
    main()
