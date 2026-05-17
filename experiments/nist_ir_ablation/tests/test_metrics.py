"""Tests for metrics.py: canonicalization and Top-k accuracy."""

from experiments.nist_ir_ablation.transpec_ext.metrics import (
    canonicalize_smiles,
    compute_top_k_accuracy,
    compute_all_metrics,
)


# ── canonicalize_smiles ────────────────────────────────────────────────


def test_canonicalize_valid_smiles():
    """Canonicalize a standard organic SMILES."""
    result = canonicalize_smiles("CCO")
    assert result == "CCO"
    assert isinstance(result, str)


def test_canonicalize_equivalent_smiles():
    """Equivalent SMILES representations produce the same canonical form."""
    can1 = canonicalize_smiles("CCO")
    can2 = canonicalize_smiles("OCC")
    assert can1 == can2 == "CCO"


def test_canonicalize_ethanol_variants():
    """Ethanol written in different orders."""
    assert canonicalize_smiles("CCO") == "CCO"
    assert canonicalize_smiles("OCC") == "CCO"
    assert canonicalize_smiles("C(C)O") == "CCO"


def test_canonicalize_benzene():
    """Benzene canonical forms."""
    assert canonicalize_smiles("c1ccccc1") == "c1ccccc1"


def test_canonicalize_invalid_smiles_returns_none():
    """Invalid SMILES returns None."""
    # Note: RDKit may treat empty string as an empty molecule,
    # so use clearly invalid SMILES.
    assert canonicalize_smiles("XXX") is None
    assert canonicalize_smiles("C1XX") is None
    assert canonicalize_smiles("!!invalid!!") is None
    assert canonicalize_smiles("C") is not None  # single atom is valid


# ── compute_top_k_accuracy ─────────────────────────────────────────────


def test_top1_hit():
    """Correct candidate at rank 1."""
    result = compute_top_k_accuracy("CCO", ["CCO", "CCC", "C=O"], [1, 3, 5])
    assert result["hit_top1"] is True
    assert result["hit_top3"] is True
    assert result["hit_top5"] is True


def test_top3_hit_not_top1():
    """Correct candidate at rank 2 — not a top-1 hit, but a top-3 hit."""
    result = compute_top_k_accuracy("CCO", ["CCC", "CCO", "C=O"], [1, 3])
    assert result["hit_top1"] is False
    assert result["hit_top3"] is True


def test_top5_hit_not_top3():
    """Correct candidate at rank 4 — only a top-5 hit."""
    result = compute_top_k_accuracy(
        "CCO",
        ["CCC", "C#N", "C=O", "CCO", "c1ccccc1"],
        [1, 3, 5, 10],
    )
    assert result["hit_top1"] is False
    assert result["hit_top3"] is False
    assert result["hit_top5"] is True
    assert result["hit_top10"] is True


def test_no_hit():
    """Correct candidate not in top-k."""
    result = compute_top_k_accuracy("CCO", ["CCC", "C#N", "C=O"], [1, 3])
    assert result["hit_top1"] is False
    assert result["hit_top3"] is False


def test_fewer_than_10_candidates():
    """Only 3 candidates — missing ranks count as misses."""
    result = compute_top_k_accuracy(
        "CCO", ["CCC", "C#N", "C=O"], [1, 3, 5, 10]
    )
    assert result["hit_top1"] is False
    assert result["hit_top3"] is False
    assert result["hit_top5"] is False
    assert result["hit_top10"] is False


def test_invalid_prediction_counts_as_miss():
    """An invalid candidate SMILES cannot match and counts as a miss."""
    result = compute_top_k_accuracy("CCO", ["XXXinvalid", "CCO"], [1, 3])
    assert result["hit_top1"] is False  # rank 1 is invalid
    assert result["hit_top3"] is True   # rank 2 is a hit


def test_invalid_label_returns_all_false():
    """If the label itself is invalid, all top-k are false."""
    result = compute_top_k_accuracy("XXXX", ["CCO", "CCC"], [1, 3, 5])
    assert all(not result[f"hit_top{k}"] for k in [1, 3, 5])


def test_empty_candidates():
    """Empty candidate list — all top-k are false."""
    result = compute_top_k_accuracy("CCO", [], [1, 3, 5])
    assert result["hit_top1"] is False
    assert result["hit_top3"] is False


def test_preserves_ranking_with_invalid_candidates():
    """Invalid predictions at early ranks block hits at later ranks for small k."""
    # Candidates: invalid, invalid, CCO
    result = compute_top_k_accuracy("CCO", ["", "XXX", "CCO"], [1, 3, 5])
    assert result["hit_top1"] is False   # rank 1 is empty string
    assert result["hit_top3"] is True    # rank 3 matches
    assert result["hit_top5"] is True


# ── compute_all_metrics ────────────────────────────────────────────────


def test_compute_all_metrics_basic():
    """Aggregate metrics across multiple examples."""
    labels = ["CCO", "CCC", "CCN"]
    candidates = [
        ["CCO", "CCC", "CCN"],        # top-1 hit (CCO)
        ["CCC", "CCO", "CCN"],        # top-1 hit (CCC)
        ["CCO", "CCC", "CCN"],        # top-1 = miss, top-3 = hit
    ]
    result = compute_all_metrics(labels, candidates, [1, 3])
    assert result["top1"] == 2.0 / 3.0  # examples 0 and 1 hit top-1
    assert result["top3"] == 1.0         # all hit top-3
