#!/usr/bin/env python3
"""
Prepare JSONL IR data for NIST ablation experiments.

Reads IR_nist.jsonl (or subset), validates, canonicalizes SMILES,
resamples spectra, splits into train/valid/test, builds tokenizer
vocabularies, and saves processed artifacts.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

# Add project root and package to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.nist_ir_ablation.transpec_ext.data_jsonl import read_jsonl
from experiments.nist_ir_ablation.transpec_ext.io_utils import (
    make_processed_dirs,
    save_json,
    save_torch,
    append_jsonl,
)
from experiments.nist_ir_ablation.transpec_ext.spectrum_preprocess import (
    preprocess_record,
)
from experiments.nist_ir_ablation.transpec_ext.tokenizer_atom import AtomTokenizer
from experiments.nist_ir_ablation.transpec_ext.tokenizer_spe import SPETokenizer
from experiments.nist_ir_ablation.transpec_ext.vocab import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL IR data for NIST ablation experiments"
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--output", required=True, help="Output root directory (e.g., runs/nist_ir_ablation_smoke)"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing processed data"
    )
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def canonicalize_smiles(smiles):
    """Canonicalize a SMILES string using RDKit. Returns (canonical, error_reason)."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "RDKit MolFromSmiles returned None"
        canonical = Chem.MolToSmiles(mol)
        return canonical, None
    except Exception as e:
        return None, str(e)


def compute_smiles_stats(sequences, max_len):
    """Compute token-length statistics for a list of ID sequences."""
    lengths = [len(seq) for seq in sequences]
    arr = np.array(lengths)
    num_truncated = int((arr > max_len).sum())
    return {
        "max_length": int(arr.max()) if len(arr) > 0 else 0,
        "min_length": int(arr.min()) if len(arr) > 0 else 0,
        "mean_length": float(arr.mean()) if len(arr) > 0 else 0.0,
        "p95_length": float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0,
        "p99_length": float(np.percentile(arr, 99)) if len(arr) > 0 else 0.0,
        "num_truncated": num_truncated,
        "max_allowed_len": max_len,
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    # Extract config sections with defaults
    spectrum_cfg = config.get("spectrum", {})
    tokenizer_cfg = config.get("tokenizer", {})
    spe_cfg = config.get("spe", {})
    split_cfg = config.get("split", {})

    n_points = spectrum_cfg.get("target_len", 3000)
    x_min = spectrum_cfg.get("x_min", 552.0)
    x_max = spectrum_cfg.get("x_max", 3844.0)
    clip_negative = spectrum_cfg.get("clip_negative", True)
    normalize_mode = spectrum_cfg.get("normalize", "max")

    max_len = tokenizer_cfg.get("max_len", 256)
    spe_vocab_size = spe_cfg.get("vocab_size", 100)
    spe_min_frequency = spe_cfg.get("min_frequency", 2)

    train_ratio = split_cfg.get("train", 0.8)
    valid_ratio = split_cfg.get("valid", 0.1)
    test_ratio = split_cfg.get("test", 0.1)
    split_seed = split_cfg.get("split_seed", 1234)

    # Override with data section if present
    data_cfg = config.get("data", {})
    if "split_seed" in data_cfg:
        split_seed = data_cfg["split_seed"]
    if "train_ratio" in data_cfg:
        train_ratio = data_cfg["train_ratio"]
    if "valid_ratio" in data_cfg:
        valid_ratio = data_cfg["valid_ratio"]
    if "test_ratio" in data_cfg:
        test_ratio = data_cfg["test_ratio"]

    # Build output paths
    output_root = os.path.abspath(args.output)
    common_dir, atom_dir, spe_dir = make_processed_dirs(output_root)

    split_indices_path = os.path.join(common_dir, "split_indices.json")
    records_path = os.path.join(common_dir, "records.pt")
    manifest_path = os.path.join(common_dir, "preprocess_manifest.json")
    smiles_stats_path = os.path.join(common_dir, "smiles_stats.json")
    spectrum_stats_path = os.path.join(common_dir, "spectrum_stats.json")
    invalid_path = os.path.join(common_dir, "invalid_records.jsonl")

    print(f"Output root: {output_root}")
    print(f"Config: {args.config}")
    print(f"Input: {args.input}")
    print()

    manifest = {
        "input_file": args.input,
        "config_file": args.config,
        "output_root": output_root,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "n_points": n_points,
            "x_min": x_min,
            "x_max": x_max,
            "clip_negative": clip_negative,
            "normalize": normalize_mode,
            "max_len": max_len,
            "spe_vocab_size": spe_vocab_size,
            "spe_min_frequency": spe_min_frequency,
            "split_seed": split_seed,
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
        },
    }

    # =========================================================
    # Step 1: Read and validate JSONL
    # =========================================================
    print("=" * 60)
    print("Step 1: Reading JSONL...")
    records, invalid_entries = read_jsonl(args.input)

    # Write invalid entries (ensure file exists even if empty)
    for entry in invalid_entries:
        append_jsonl(entry, invalid_path)
    if not os.path.exists(invalid_path):
        # Create empty file
        append_jsonl({}, invalid_path)
        # Remove the dummy entry so the file is truly empty
        with open(invalid_path, "w") as f:
            pass

    print(f"  Valid records: {len(records)}")
    print(f"  Invalid entries: {len(invalid_entries)}")
    manifest["num_input_records"] = len(records)
    manifest["num_invalid_entries"] = len(invalid_entries)

    if len(records) == 0:
        print("ERROR: No valid records found. Exiting.")
        save_json(manifest, manifest_path)
        sys.exit(1)

    # =========================================================
    # Step 2: Canonicalize SMILES
    # =========================================================
    print("=" * 60)
    print("Step 2: Canonicalizing SMILES...")

    canonical_records = []
    for record in records:
        canonical, error = canonicalize_smiles(record.smiles)
        if canonical is None:
            append_jsonl(
                {
                    "line_number": record.idx + 1,
                    "reason": f"Invalid SMILES: {error}",
                    "smiles": record.smiles,
                },
                invalid_path,
            )
        else:
            canonical_records.append((canonical, record))

    print(f"  Valid after canonicalization: {len(canonical_records)}")
    manifest["num_valid_after_canonicalization"] = len(canonical_records)

    if len(canonical_records) == 0:
        print("ERROR: No valid records after SMILES canonicalization. Exiting.")
        save_json(manifest, manifest_path)
        sys.exit(1)

    # =========================================================
    # Step 3: Preprocess spectra
    # =========================================================
    print("=" * 60)
    print("Step 3: Preprocessing spectra...")

    processed = []
    spectrum_failures = 0
    raw_intensities = []
    raw_wavenumbers = []

    for canonical, record in canonical_records:
        spectrum = preprocess_record(
            record.wavenumbers,
            record.intensities,
            n_points=n_points,
            low=x_min,
            high=x_max,
            normalize=normalize_mode,
            clip_negative=clip_negative,
        )
        if spectrum is None:
            append_jsonl(
                {
                    "line_number": record.idx + 1,
                    "reason": "Spectrum preprocessing failed (NaN/inf/zero)",
                    "smiles": record.smiles,
                },
                invalid_path,
            )
            spectrum_failures += 1
        else:
            processed.append((canonical, record, spectrum))
            if len(raw_intensities) < 10000:  # sample for stats
                raw_intensities.extend(record.intensities)
                raw_wavenumbers.extend(record.wavenumbers)

    print(f"  Processed successfully: {len(processed)}")
    print(f"  Spectrum failures: {spectrum_failures}")
    manifest["num_processed"] = len(processed)
    manifest["num_spectrum_failures"] = spectrum_failures

    if len(processed) == 0:
        print("ERROR: No records passed preprocessing. Exiting.")
        save_json(manifest, manifest_path)
        sys.exit(1)

    # Spectrum stats
    raw_arr = np.array(raw_intensities)
    wavenum_arr = np.array(raw_wavenumbers)
    spectrum_stats = {
        "raw_intensity_min": float(raw_arr.min()) if len(raw_arr) > 0 else 0,
        "raw_intensity_max": float(raw_arr.max()) if len(raw_arr) > 0 else 0,
        "raw_intensity_mean": float(raw_arr.mean()) if len(raw_arr) > 0 else 0,
        "raw_wavenumber_min": float(wavenum_arr.min()) if len(wavenum_arr) > 0 else 0,
        "raw_wavenumber_max": float(wavenum_arr.max()) if len(wavenum_arr) > 0 else 0,
        "target_len": n_points,
        "target_x_min": x_min,
        "target_x_max": x_max,
    }
    save_json(spectrum_stats, spectrum_stats_path)
    print(f"  Spectrum stats saved to {spectrum_stats_path}")

    # =========================================================
    # Step 4: Save common records
    # =========================================================
    print("=" * 60)
    print("Step 4: Saving common records...")

    canonicals = [p[0] for p in processed]
    raw_smiles_list = [p[1].smiles for p in processed]
    spectra = np.stack([p[2] for p in processed]).astype(np.float32)
    indices = [p[1].idx for p in processed]

    records_data = {
        "canonical_smiles": canonicals,
        "raw_smiles": raw_smiles_list,
        "spectrum": torch.from_numpy(spectra),  # (N, 3000)
        "index": indices,
        "metadata": {
            "n_points": n_points,
            "x_min": x_min,
            "x_max": x_max,
        },
    }
    save_torch(records_data, records_path)
    print(f"  Records saved: {len(canonicals)} spectra to {records_path}")
    print(f"  Spectrum tensor shape: {spectra.shape}")

    # =========================================================
    # Step 5: Create or reuse split indices
    # =========================================================
    print("=" * 60)
    print("Step 5: Creating train/valid/test split...")

    n_total = len(processed)

    if os.path.exists(split_indices_path) and not args.force:
        with open(split_indices_path) as f:
            split_data = json.load(f)
        train_idx = split_data["train"]
        valid_idx = split_data["valid"]
        test_idx = split_data["test"]
        print(f"  Reusing existing split from {split_indices_path}")
    else:
        rng = np.random.default_rng(split_seed)
        all_idx = np.arange(n_total)
        rng.shuffle(all_idx)

        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)

        train_idx = sorted(all_idx[:n_train].tolist())
        valid_idx = sorted(all_idx[n_train: n_train + n_valid].tolist())
        test_idx = sorted(all_idx[n_train + n_valid:].tolist())

        split_data = {
            "train": train_idx,
            "valid": valid_idx,
            "test": test_idx,
            "split_seed": split_seed,
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "n_total": n_total,
        }
        save_json(split_data, split_indices_path)
        print(f"  New split saved to {split_indices_path}")

    print(f"  Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
    manifest["split"] = {
        "train": len(train_idx),
        "valid": len(valid_idx),
        "test": len(test_idx),
        "split_seed": split_seed,
        "split_file": split_indices_path,
    }

    # =========================================================
    # Step 6: Build atom tokenizer and encode datasets
    # =========================================================
    print("=" * 60)
    print("Step 6: Building atom tokenizer...")

    train_smiles = [canonicals[i] for i in train_idx]
    valid_smiles = [canonicals[i] for i in valid_idx]
    test_smiles = [canonicals[i] for i in test_idx]

    atom_tokenizer = AtomTokenizer(max_len=max_len)
    atom_tokenizer.fit(train_smiles)

    atom_vocab_path = os.path.join(atom_dir, "atom_vocab.json")
    atom_tokenizer.save(atom_vocab_path)
    print(f"  Atom vocab size: {atom_tokenizer.vocab_size}")
    print(f"  Atom vocab saved to {atom_vocab_path}")

    # Encode splits
    for split_name, split_smiles, split_idx in [
        ("train", train_smiles, train_idx),
        ("valid", valid_smiles, valid_idx),
        ("test", test_smiles, test_idx),
    ]:
        encoded = []
        for s in split_smiles:
            ids = atom_tokenizer.encode(s, add_special_tokens=True, max_len=max_len)
            encoded.append(ids)
        save_torch(encoded, os.path.join(atom_dir, f"{split_name}.pt"))
        print(f"  atom/{split_name}.pt: {len(encoded)} sequences")

    # Compute atom token length stats (on full set for reference)
    atom_stats = compute_smiles_stats(
        [atom_tokenizer.encode(s, add_special_tokens=True, max_len=max_len) for s in train_smiles],
        max_len,
    )
    manifest["atom_smiles_stats"] = atom_stats
    save_json(atom_stats, smiles_stats_path)

    if atom_stats["num_truncated"] > 0:
        print(
            f"  WARNING: {atom_stats['num_truncated']} train sequences truncated "
            f"(max_len={max_len})!"
        )
        manifest["warning_truncated"] = (
            f"{atom_stats['num_truncated']} train sequences truncated at max_len={max_len}"
        )

    # =========================================================
    # Step 7: Build SPE tokenizer and encode datasets
    # =========================================================
    print("=" * 60)
    print("Step 7: Building SPE tokenizer...")

    spe_tokenizer = SPETokenizer(
        vocab_size=spe_vocab_size,
        min_frequency=spe_min_frequency,
        max_len=max_len,
    )
    spe_tokenizer.fit(train_smiles)

    spe_vocab_path = os.path.join(spe_dir, "spe_vocab.json")
    spe_merges_path_txt = os.path.join(spe_dir, "spe_merges.txt")
    spe_tokenizer.save(os.path.join(spe_dir, "spe").rstrip("/"))
    print(f"  SPE vocab size: {spe_tokenizer.vocab_size_real}")
    print(f"  SPE merges: {len(spe_tokenizer.merges)}")
    print(f"  SPE vocab saved to {spe_vocab_path}")
    print(f"  SPE merges saved to {spe_merges_path_txt}")

    # Encode splits
    for split_name, split_smiles, split_idx in [
        ("train", train_smiles, train_idx),
        ("valid", valid_smiles, valid_idx),
        ("test", test_smiles, test_idx),
    ]:
        encoded = []
        for s in split_smiles:
            ids = spe_tokenizer.encode(s, add_special_tokens=True, max_len=max_len)
            encoded.append(ids)
        save_torch(encoded, os.path.join(spe_dir, f"{split_name}.pt"))
        print(f"  spe/{split_name}.pt: {len(encoded)} sequences")

    # =========================================================
    # Step 8: Save manifest and print summary
    # =========================================================
    manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["status"] = "completed"
    save_json(manifest, manifest_path)

    print()
    print("=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Total records processed: {len(processed)}")
    print(f"  Invalid records: {len(invalid_entries) + spectrum_failures}")
    print(f"  Train: {len(train_idx)} | Valid: {len(valid_idx)} | Test: {len(test_idx)}")
    print(f"  Atom vocab size: {atom_tokenizer.vocab_size}")
    print(f"  SPE vocab size: {spe_tokenizer.vocab_size_real}")
    print(f"  SPE merge rules: {len(spe_tokenizer.merges)}")
    print(f"  Output: {output_root}")
    print(f"  Manifest: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    import torch

    main()
