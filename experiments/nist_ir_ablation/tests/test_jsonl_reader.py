"""Tests for data_jsonl.py"""
import json
import tempfile
import os

from experiments.nist_ir_ablation.transpec_ext.data_jsonl import read_jsonl


def _make_jsonl(lines):
    """Write a temporary JSONL file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for line in lines:
        tmp.write(line + "\n")
    tmp.close()
    return tmp.name


def test_valid_record():
    lines = [
        json.dumps({
            "smiles": "CCO",
            "value": {"x": [1000.0, 2000.0], "y": [0.1, 0.2]},
        })
    ]
    path = _make_jsonl(lines)
    records, invalid = read_jsonl(path, min_points=1)
    os.unlink(path)

    assert len(records) == 1
    assert len(invalid) == 0
    assert records[0].smiles == "CCO"
    assert len(records[0].wavenumbers) == 2
    assert len(records[0].intensities) == 2


def test_invalid_json_line():
    lines = [
        '{"smiles": "CCO", "value": {"x": [1], "y": [2]}}',
        "not valid json at all",
    ]
    path = _make_jsonl(lines)
    records, invalid = read_jsonl(path, min_points=1)
    os.unlink(path)

    assert len(records) == 1
    assert len(invalid) == 1
    assert "Invalid JSON" in invalid[0]["reason"]


def test_invalid_smiles_handled_gracefully():
    """Invalid SMILES are NOT rejected by data_jsonl — that happens during canonicalization."""
    lines = [
        json.dumps({
            "smiles": "CCO",
            "value": {"x": [1000.0, 2000.0], "y": [0.1, 0.2]},
        })
    ]
    path = _make_jsonl(lines)
    records, invalid = read_jsonl(path, min_points=1)
    os.unlink(path)

    # data_jsonl does not validate SMILES content, only field presence
    assert len(records) == 1
    assert len(invalid) == 0


def test_missing_fields():
    lines = [
        json.dumps({"smiles": "CCO"}),  # missing value
        json.dumps({"value": {"x": [1], "y": [2]}}),  # missing smiles
    ]
    path = _make_jsonl(lines)
    records, invalid = read_jsonl(path, min_points=1)
    os.unlink(path)

    assert len(records) == 0
    assert len(invalid) == 2


def test_mismatched_x_y_length():
    lines = [
        json.dumps({
            "smiles": "CCO",
            "value": {"x": [1.0, 2.0, 3.0], "y": [0.1]},
        })
    ]
    path = _make_jsonl(lines)
    records, invalid = read_jsonl(path, min_points=1)
    os.unlink(path)

    assert len(records) == 0
    assert len(invalid) == 1
    assert "Mismatched" in invalid[0]["reason"]


def test_below_min_points():
    lines = [
        json.dumps({
            "smiles": "CCO",
            "value": {"x": [1.0], "y": [0.1]},
        })
    ]
    path = _make_jsonl(lines)
    records, invalid = read_jsonl(path, min_points=5)
    os.unlink(path)

    assert len(records) == 0
    assert len(invalid) == 1
    assert "Too few points" in invalid[0]["reason"]


def test_empty_lines_skipped():
    lines = [
        "",
        json.dumps({
            "smiles": "CCO",
            "value": {"x": [1000.0], "y": [0.1]},
        }),
        "",
    ]
    path = _make_jsonl(lines)
    records, invalid = read_jsonl(path, min_points=1)
    os.unlink(path)

    assert len(records) == 1
    assert len(invalid) == 0
