#!/usr/bin/env python3
"""Evaluate a single trained NIST IR ablation model on the test set.

Runs threshold-value decoding and computes Top-1/3/5/10 accuracy.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for p in [_PROJECT_ROOT, _SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.nist_ir_ablation.transpec_ext.dataset import (
    NistIRAblationDataset,
    load_split_dataset,
)
from experiments.nist_ir_ablation.transpec_ext.decoding import decode_ablation
from experiments.nist_ir_ablation.transpec_ext.metrics import (
    canonicalize_smiles,
    compute_all_metrics,
    compute_top_k_accuracy,
)
from experiments.nist_ir_ablation.transpec_ext.model_factory import build_model
from experiments.nist_ir_ablation.transpec_ext.tokenizer_atom import AtomTokenizer
from experiments.nist_ir_ablation.transpec_ext.tokenizer_spe import SPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        description="Evaluate a single NIST IR ablation model",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to condition-specific YAML (e.g., E0_atom_nope.yaml)",
    )
    parser.add_argument(
        "--processed_dir",
        required=True,
        help="Path to processed data root",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint .pt file (e.g., .../checkpoints/best.pt)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g., 'cuda:0' or 'cpu')",
    )
    # Decoding overrides
    parser.add_argument(
        "--max_decode_len", type=int, default=None,
    )
    parser.add_argument(
        "--beam_size", type=int, default=None,
    )
    parser.add_argument(
        "--threshold_value", type=float, default=None,
    )
    parser.add_argument(
        "--candidate_limit", type=int, default=None,
    )
    parser.add_argument(
        "--decode_method", default=None,
        choices=["beam", "threshold"],
        help="Decoding method (default: beam)",
    )
    parser.add_argument(
        "--base_config", default=None,
        help="Base YAML config path (default: smoke_200.yaml next to condition config)",
    )
    return parser.parse_args()


def load_configs(args):
    """Load and merge condition config over base config."""
    base_path = args.base_config
    if base_path is None:
        config_dir = os.path.dirname(os.path.abspath(args.config))
        base_path = os.path.join(config_dir, "smoke_200.yaml")
    if not os.path.exists(base_path):
        print(f"WARNING: Base config not found at {base_path}, using condition config only")
        with open(args.config) as f:
            return yaml.safe_load(f)

    with open(base_path) as f:
        config = yaml.safe_load(f)
    with open(args.config) as f:
        cond_cfg = yaml.safe_load(f)

    config = deep_update(config, cond_cfg)
    return config


def load_tokenizer(processed_dir: str, tokenizer_type: str, max_len: int = 256):
    """Load tokenizer from processed data directory."""
    if tokenizer_type == "spe":
        # The processed_dir/spe/ directory should contain spe_vocab.json and spe_merges.txt
        # SPETokenizer.load expects a stem path and appends _vocab.json / _merges.txt
        spe_stem = os.path.join(processed_dir, "spe", "spe")
        tokenizer = SPETokenizer.load(spe_stem)
    else:
        vocab_path = os.path.join(processed_dir, "atom", "atom_vocab.json")
        tokenizer = AtomTokenizer.load(vocab_path)
        tokenizer.max_len = max_len
    return tokenizer


def decode_dataset(
    model,
    dataset: NistIRAblationDataset,
    tokenizer,
    device: torch.device,
    decoding_cfg: dict,
):
    """Decode all examples in the dataset.

    Returns:
        all_labels: list of label SMILES strings
        all_candidates: list of list of candidate SMILES strings
        all_scores: list of list of candidate scores
        all_invalid: list of invalid decode records
    """
    all_labels = []
    all_candidates = []
    all_scores = []
    all_invalid = []

    decode_method = decoding_cfg.get("decode_method", "beam")
    beam_size = decoding_cfg.get("beam_size", 10)
    candidate_limit = decoding_cfg.get("candidate_limit", 10)
    max_decode_len = decoding_cfg.get("max_len", 256)
    threshold_value = decoding_cfg.get("threshold_value", 0.01)

    model.eval()

    for i in range(len(dataset)):
        sample = dataset[i]
        spectrum = sample["spectrum"].unsqueeze(0).to(device)  # (1, 3000)
        label_smi = sample.get("canonical_smiles", sample.get("raw_smiles", ""))

        decode_kwargs = {
            "max_len": max_decode_len,
            "candidate_limit": candidate_limit,
            "decode_method": decode_method,
        }
        if decode_method == "beam":
            decode_kwargs["beam_size"] = beam_size
        else:
            decode_kwargs["threshold_value"] = threshold_value

        with torch.no_grad():
            try:
                candidates, scores, _ = decode_ablation(
                    model,
                    spectrum,
                    tokenizer,
                    **decode_kwargs,
                )
            except Exception as exc:
                logger.error("Decoding failed for idx=%s: %s", sample.get("idx", i), exc)
                candidates = []
                scores = []

        # Record invalid decodes
        for rank, smi in enumerate(candidates):
            if canonicalize_smiles(smi) is None:
                all_invalid.append({
                    "idx": sample.get("idx", i),
                    "rank": rank + 1,
                    "raw_smiles": smi,
                    "reason": "invalid SMILES",
                })

        all_labels.append(label_smi)
        all_candidates.append(candidates)
        all_scores.append(scores)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "  decoded %d/%d (candidates: %d)",
                i + 1, len(dataset), len(candidates),
            )

    return all_labels, all_candidates, all_scores, all_invalid


def save_outputs(
    output_dir: str,
    model_id: str,
    seed: int,
    dataset_indices,
    label_smiles_list,
    candidates_list,
    scores_list,
    invalid_records,
    top_k_values,
    metrics,
    checkpoint_path,
    processed_dir,
    eval_time_sec,
):
    """Save all evaluation output files."""
    os.makedirs(output_dir, exist_ok=True)

    # metrics.json
    metrics_data = {
        "model_id": model_id,
        "seed": seed,
        "num_test": metrics.get("num_test", len(label_smiles_list)),
        "top1": metrics.get("top1", 0.0),
        "top3": metrics.get("top3", 0.0),
        "top5": metrics.get("top5", 0.0),
        "top10": metrics.get("top10", 0.0),
        "invalid_decode_count": metrics.get("invalid_decode_count", 0),
        "invalid_decode_rate": metrics.get("invalid_decode_rate", 0.0),
        "avg_num_candidates": metrics.get("avg_num_candidates", 0.0),
        "decode_method": metrics.get("decode_method", ""),
        "beam_size": metrics.get("beam_size", 0),
        "candidate_limit": metrics.get("candidate_limit", 0),
        "max_decode_len": metrics.get("max_decode_len", 0),
        "checkpoint_path": checkpoint_path,
        "processed_dir": processed_dir,
        "eval_time_sec": eval_time_sec,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info("Saved %s", metrics_path)

    # predictions_topk.csv
    top_k = sorted(top_k_values)
    pred_cols = [f"pred_{k}" for k in range(1, max(top_k) + 1)]
    hit_cols = [f"hit_top{k}" for k in top_k]
    fieldnames = ["idx", "label_smiles"] + pred_cols + hit_cols

    csv_path = os.path.join(output_dir, "predictions_topk.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(label_smiles_list)):
            row = {
                "idx": dataset_indices[i] if dataset_indices else i,
                "label_smiles": label_smiles_list[i],
            }
            cands = candidates_list[i] if i < len(candidates_list) else []
            for rank in range(1, max(top_k) + 1):
                rank_idx = rank - 1
                if rank_idx < len(cands):
                    row[f"pred_{rank}"] = cands[rank_idx]
                else:
                    row[f"pred_{rank}"] = ""
            # Compute hits for this row
            result = compute_top_k_accuracy(
                label_smiles_list[i], cands, top_k_values
            )
            for k in top_k:
                row[f"hit_top{k}"] = result.get(f"hit_top{k}", False)
            writer.writerow(row)
    logger.info("Saved %s", csv_path)

    # candidates.jsonl
    cand_path = os.path.join(output_dir, "candidates.jsonl")
    with open(cand_path, "w") as f:
        for i in range(len(label_smiles_list)):
            cands = candidates_list[i] if i < len(candidates_list) else []
            scs = scores_list[i] if i < len(scores_list) else []
            result = compute_top_k_accuracy(
                label_smiles_list[i], cands, top_k_values
            )
            entry = {
                "idx": dataset_indices[i] if dataset_indices else i,
                "label_smiles": label_smiles_list[i],
                "candidates": [
                    {"rank": r + 1, "smiles": cands[r], "score": float(scs[r])}
                    for r in range(min(len(cands), len(scs)))
                ],
            }
            for k in top_k:
                entry[f"hit_top{k}"] = result.get(f"hit_top{k}", False)
            f.write(json.dumps(entry) + "\n")
    logger.info("Saved %s", cand_path)

    # invalid_decodes.jsonl
    inv_path = os.path.join(output_dir, "invalid_decodes.jsonl")
    with open(inv_path, "w") as f:
        for rec in invalid_records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Saved %s (entries: %d)", inv_path, len(invalid_records))


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load config
    config = load_configs(args)

    model_id = os.path.splitext(os.path.basename(args.config))[0]
    use_fourier = config.get("spectral_fourier_encoding", False)
    tokenizer_type = config.get("tokenizer", {}).get("type", "atom")
    decoding_cfg = config.get("decoding", {})
    top_k_values = decoding_cfg.get("topk", [1, 3, 5, 10])

    # CLI overrides for decoding
    if args.max_decode_len is not None:
        decoding_cfg["max_len"] = args.max_decode_len
    if args.threshold_value is not None:
        decoding_cfg["threshold_value"] = args.threshold_value
    if args.candidate_limit is not None:
        decoding_cfg["candidate_limit"] = args.candidate_limit
    if args.beam_size is not None:
        decoding_cfg["beam_size"] = args.beam_size
    if args.decode_method is not None:
        decoding_cfg["decode_method"] = args.decode_method

    logger.info("Model: %s", model_id)
    logger.info("Fourier: %s", use_fourier)
    logger.info("Tokenizer: %s", tokenizer_type)

    # Load tokenizer
    tokenizer = load_tokenizer(args.processed_dir, tokenizer_type)
    vocab_size = tokenizer.vocab_size
    logger.info("Vocab size: %d", vocab_size)

    # Load test dataset
    logger.info("Loading test dataset from %s ...", args.processed_dir)
    target_dir = "spe" if tokenizer_type == "spe" else "atom"
    test_dataset = load_split_dataset(args.processed_dir, "test", target_dir=target_dir)
    num_test = len(test_dataset)
    logger.info("Test samples: %d", num_test)

    if num_test == 0:
        logger.warning("Test dataset is empty, nothing to evaluate.")
        return

    # Build model
    logger.info("Building model ...")
    model = build_model(config, vocab_size)
    model = model.to(device)

    # Load checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try loading directly as state dict
        model.load_state_dict(checkpoint)
    logger.info("Checkpoint loaded")

    # Decode
    logger.info("Decoding test set ...")
    t0 = time.time()

    all_labels, all_candidates, all_scores, all_invalid = decode_dataset(
        model=model,
        dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        decoding_cfg=decoding_cfg,
    )

    eval_time = time.time() - t0
    logger.info("Decoding complete in %.1f sec", eval_time)

    # Compute metrics
    top_k_results = compute_all_metrics(all_labels, all_candidates, top_k_values)

    # Compute additional stats
    num_invalid = len(all_invalid)
    total_candidates = sum(len(c) for c in all_candidates)
    avg_candidates = total_candidates / num_test if num_test > 0 else 0

    metrics = {
        "num_test": num_test,
        "top1": top_k_results.get("top1", 0.0),
        "top3": top_k_results.get("top3", 0.0),
        "top5": top_k_results.get("top5", 0.0),
        "top10": top_k_results.get("top10", 0.0),
        "invalid_decode_count": num_invalid,
        "invalid_decode_rate": num_invalid / total_candidates if total_candidates > 0 else 0.0,
        "avg_num_candidates": avg_candidates,
        "decode_method": decoding_cfg.get("decode_method", "beam"),
        "beam_size": decoding_cfg.get("beam_size", 10),
        "candidate_limit": decoding_cfg.get("candidate_limit", 10),
        "max_decode_len": decoding_cfg.get("max_len", 256),
        "checkpoint_path": args.checkpoint,
        "processed_dir": args.processed_dir,
        "eval_time_sec": eval_time,
    }

    logger.info("Results:")
    for k in [1, 3, 5, 10]:
        logger.info("  Top-%d: %.4f", k, metrics.get(f"top{k}", 0.0))
    logger.info("  Invalid decodes: %d (%.2f%%)", num_invalid,
                100.0 * metrics["invalid_decode_rate"])
    logger.info("  Avg candidates: %.1f", avg_candidates)

    # Save outputs
    dataset_indices = [test_dataset[i]["idx"] for i in range(len(test_dataset))]
    save_outputs(
        output_dir=args.output,
        model_id=model_id,
        seed=args.seed,
        dataset_indices=dataset_indices,
        label_smiles_list=all_labels,
        candidates_list=all_candidates,
        scores_list=all_scores,
        invalid_records=all_invalid,
        top_k_values=top_k_values,
        metrics=metrics,
        checkpoint_path=args.checkpoint,
        processed_dir=args.processed_dir,
        eval_time_sec=eval_time,
    )

    logger.info("Evaluation complete. Output: %s", args.output)


if __name__ == "__main__":
    main()
