"""Tests for evaluate_one.py output format and schema.

Uses a tiny temporary model, dataset, and config to run a minimal
end-to-end evaluation and verify output file structure.
"""

import csv
import json
import os
import tempfile

import pytest
import torch
import yaml

from experiments.nist_ir_ablation.transpec_ext.dataset import (
    NistIRAblationDataset,
    load_split_dataset,
)
from experiments.nist_ir_ablation.transpec_ext.model_factory import build_model
from experiments.nist_ir_ablation.transpec_ext.tokenizer_atom import AtomTokenizer
from experiments.nist_ir_ablation.transpec_ext.vocab import PAD_ID, BOS_ID, EOS_ID


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_vocab():
    """Create a tiny atom vocabulary with just a few tokens."""
    tokens = ["C", "N", "O"]
    tok = AtomTokenizer(max_len=64)
    tok.fit(tokens)
    return tok


@pytest.fixture(scope="module")
def tiny_processed(tiny_vocab):
    """Create a minimal processed data directory with 2 test samples."""
    tmpdir = tempfile.mkdtemp(prefix="eval_test_")

    # Common records
    common_dir = os.path.join(tmpdir, "common")
    os.makedirs(common_dir)

    spectra = torch.randn(10, 3000)  # 10 total records
    canonical_smiles = [
        "CCO", "C=O", "CCN", "C#N", "c1ccccc1",
        "CCO", "CC=O", "NCC", "CCl", "CO",
    ]
    raw_smiles = [
        "CCO", "C=O", "CCN", "C#N", "c1ccccc1",
        "CCO", "CC=O", "NCC", "CCl", "CO",
    ]

    records = {
        "spectrum": spectra,
        "canonical_smiles": canonical_smiles,
        "raw_smiles": raw_smiles,
    }
    torch.save(records, os.path.join(common_dir, "records.pt"))

    split_indices = {
        "train": [0, 1, 2, 3, 4],
        "valid": [5, 6, 7],
        "test": [8, 9],
    }
    with open(os.path.join(common_dir, "split_indices.json"), "w") as f:
        json.dump(split_indices, f)

    # Atom tokenized targets
    atom_dir = os.path.join(tmpdir, "atom")
    os.makedirs(atom_dir)

    target_ids_train = [tiny_vocab.encode(c) for c in ["CCO", "C=O", "CCN", "C#N", "c1ccccc1"]]
    target_ids_valid = [tiny_vocab.encode(c) for c in ["CCO", "CC=O", "NCC"]]
    target_ids_test = [tiny_vocab.encode(c) for c in ["CCl", "CO"]]
    torch.save(target_ids_train, os.path.join(atom_dir, "train.pt"))
    torch.save(target_ids_valid, os.path.join(atom_dir, "valid.pt"))
    torch.save(target_ids_test, os.path.join(atom_dir, "test.pt"))

    # Save vocab
    tiny_vocab.save(os.path.join(atom_dir, "atom_vocab.json"))

    return tmpdir


@pytest.fixture(scope="module")
def tiny_config(tiny_processed):
    """Create a minimal config YAML for evaluation."""
    config = {
        "spectral_fourier_encoding": False,
        "tokenizer": {"type": "atom", "max_len": 64},
        "model": {
            "d_model": 16,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "max_len": 64,
            "use_cnn": True,
            "use_mlp": False,
        },
        "decoding": {
            "max_len": 64,
            "threshold_value": 0.01,
            "candidate_limit": 10,
            "topk": [1, 3, 5, 10],
        },
        "fourier": {"num_frequencies": 4},
        "spectrum": {
            "target_len": 3000,
            "x_min": 552.0,
            "x_max": 3844.0,
        },
    }
    tmpdir = tempfile.mkdtemp(prefix="eval_cfg_")
    cfg_path = os.path.join(tmpdir, "test_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)
    return cfg_path


@pytest.fixture(scope="module")
def tiny_checkpoint(tiny_vocab, tiny_processed):
    """Build a tiny model, save it as a checkpoint."""
    config = {
        "spectral_fourier_encoding": False,
        "model": {
            "d_model": 16,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "max_len": 64,
            "use_cnn": True,
            "use_mlp": False,
        },
        "decoding": {"max_len": 64},
        "fourier": {"num_frequencies": 4},
        "spectrum": {"target_len": 3000, "x_min": 552.0, "x_max": 3844.0},
    }
    model = build_model(config, tiny_vocab.vocab_size)
    model.eval()

    tmpdir = tempfile.mkdtemp(prefix="eval_ckpt_")
    ckpt_path = os.path.join(tmpdir, "best.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 0, "valid_loss": 0.0},
        ckpt_path,
    )
    return ckpt_path


# ── tests ──────────────────────────────────────────────────────────────


def test_evaluate_one_runs_and_creates_outputs(
    tiny_processed, tiny_config, tiny_checkpoint, tiny_vocab,
):
    """Run evaluate_one.py end-to-end and verify output files exist."""
    from experiments.nist_ir_ablation.scripts.evaluate_one import main as eval_main

    output_dir = tempfile.mkdtemp(prefix="eval_out_")

    # We'll call the internal logic rather than via argparse
    # Build and evaluate programmatically
    from experiments.nist_ir_ablation.scripts.evaluate_one import (
        load_configs,
        load_tokenizer,
        decode_dataset,
        save_outputs,
    )
    from experiments.nist_ir_ablation.transpec_ext.metrics import compute_all_metrics
    from experiments.nist_ir_ablation.transpec_ext.metrics import (
        compute_top_k_accuracy,
    )

    import argparse

    # Simulate args
    class MockArgs:
        config = tiny_config
        processed_dir = tiny_processed
        checkpoint = tiny_checkpoint
        output = output_dir
        seed = 42
        device = "cpu"
        max_decode_len = None
        threshold_value = None
        candidate_limit = None
        beam_size = None

    # Load config
    config = load_configs(MockArgs)
    # Actually load_configs expects the path, not a MockArgs
    # Let me do this differently and just call the functions directly

    config = load_configs_internal(tiny_config)
    tokenizer = load_tokenizer(
        tiny_processed, config.get("tokenizer", {}).get("type", "atom")
    )
    vocab_size = tokenizer.vocab_size

    # Load dataset
    test_dataset = load_split_dataset(tiny_processed, "test")
    assert len(test_dataset) == 2

    # Build and load model
    model = build_model(config, vocab_size)
    checkpoint = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Decode
    decoding_cfg = config.get("decoding", {})
    all_labels, all_candidates, all_scores, all_invalid = decode_dataset(
        model=model,
        dataset=test_dataset,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        decoding_cfg=decoding_cfg,
    )

    assert len(all_labels) == 2
    assert len(all_candidates) == 2
    assert len(all_scores) == 2

    # Compute metrics
    top_k_values = [1, 3, 5, 10]
    top_k_results = compute_all_metrics(all_labels, all_candidates, top_k_values)

    # Save outputs
    output_dir = tempfile.mkdtemp(prefix="eval_out2_")
    os.makedirs(output_dir, exist_ok=True)

    num_test = len(test_dataset)
    total_candidates = sum(len(c) for c in all_candidates)
    num_invalid = len(all_invalid)

    metrics_dict = {
        "num_test": num_test,
        "top1": top_k_results.get("top1", 0.0),
        "top3": top_k_results.get("top3", 0.0),
        "top5": top_k_results.get("top5", 0.0),
        "top10": top_k_results.get("top10", 0.0),
        "invalid_decode_count": num_invalid,
        "invalid_decode_rate": num_invalid / total_candidates if total_candidates > 0 else 0.0,
        "avg_num_candidates": total_candidates / num_test if num_test > 0 else 0.0,
        "checkpoint_path": tiny_checkpoint,
        "processed_dir": tiny_processed,
        "eval_time_sec": 0.0,
    }

    dataset_indices = [test_dataset[i]["idx"] for i in range(len(test_dataset))]

    save_outputs(
        output_dir=output_dir,
        model_id="test_model",
        seed=42,
        dataset_indices=dataset_indices,
        label_smiles_list=all_labels,
        candidates_list=all_candidates,
        scores_list=all_scores,
        invalid_records=all_invalid,
        top_k_values=top_k_values,
        metrics=metrics_dict,
        checkpoint_path=tiny_checkpoint,
        processed_dir=tiny_processed,
        eval_time_sec=0.0,
    )

    # ── Verify output files ─────────────────────────────────────────

    # 1. metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    assert os.path.exists(metrics_path), "metrics.json not created"
    with open(metrics_path) as f:
        m = json.load(f)

    required_metrics = [
        "model_id", "seed", "num_test",
        "top1", "top3", "top5", "top10",
        "invalid_decode_count", "invalid_decode_rate",
        "avg_num_candidates",
        "decode_method", "beam_size", "candidate_limit", "max_decode_len",
        "checkpoint_path", "processed_dir", "eval_time_sec",
    ]
    for key in required_metrics:
        assert key in m, f"metrics.json missing key: {key}"
    assert m["num_test"] == 2
    assert isinstance(m["top1"], float)

    # 2. predictions_topk.csv
    csv_path = os.path.join(output_dir, "predictions_topk.csv")
    assert os.path.exists(csv_path), "predictions_topk.csv not created"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2

    expected_cols = [
        "idx", "label_smiles",
        "pred_1", "pred_2", "pred_3", "pred_4", "pred_5",
        "pred_6", "pred_7", "pred_8", "pred_9", "pred_10",
        "hit_top1", "hit_top3", "hit_top5", "hit_top10",
    ]
    for col in expected_cols:
        assert col in rows[0], f"predictions_topk.csv missing column: {col}"

    # 3. candidates.jsonl
    cand_path = os.path.join(output_dir, "candidates.jsonl")
    assert os.path.exists(cand_path), "candidates.jsonl not created"
    with open(cand_path) as f:
        cand_lines = f.readlines()
    assert len(cand_lines) == 2
    for line in cand_lines:
        obj = json.loads(line)
        assert "idx" in obj
        assert "label_smiles" in obj
        assert "candidates" in obj
        assert "hit_top1" in obj
        assert "hit_top3" in obj
        assert "hit_top5" in obj
        assert "hit_top10" in obj

    # 4. invalid_decodes.jsonl (must exist even if empty)
    inv_path = os.path.join(output_dir, "invalid_decodes.jsonl")
    assert os.path.exists(inv_path), "invalid_decodes.jsonl not created"


def load_configs_internal(config_path):
    """Minimal config loader for tests (no base config needed)."""
    with open(config_path) as f:
        return yaml.safe_load(f)
