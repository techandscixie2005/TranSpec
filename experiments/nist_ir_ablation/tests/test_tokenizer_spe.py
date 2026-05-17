"""Tests for tokenizer_spe.py"""
import os
import tempfile

from experiments.nist_ir_ablation.transpec_ext.tokenizer_spe import SPETokenizer
from experiments.nist_ir_ablation.transpec_ext.vocab import BOS_ID, EOS_ID


def _train_spe():
    tok = SPETokenizer(vocab_size=10, min_frequency=2, max_len=256)
    tok.fit(["CCO", "CCN", "CCC", "CCO", "CCN", "CCO"])
    return tok


def test_learns_merges_from_train():
    tok = _train_spe()
    assert len(tok.merges) > 0


def test_does_not_need_valid_test_to_fit():
    tok = SPETokenizer(vocab_size=5, min_frequency=1, max_len=256)
    tok.fit(["CCO", "CCN"])
    assert tok.vocab_size_real >= 4
    # It should have learned at least some merges
    ids = tok.encode("CCO")
    assert len(ids) >= 2  # at least BOS + EOS


def test_atom_fallback():
    """Atom tokens must remain available as fallback."""
    tok = _train_spe()
    ids = tok.encode("CCO")
    decoded = tok.decode(ids)
    assert decoded == "CCO"


def test_save_load():
    tok = _train_spe()
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "spe")
        tok.save(base)

        loaded = SPETokenizer.load(base)

        assert loaded.vocab_size_real == tok.vocab_size_real
        assert len(loaded.merges) == len(tok.merges)

        for smiles in ["CCO", "CCN", "CCC"]:
            assert loaded.encode(smiles) == tok.encode(smiles)


def test_decode_returns_string():
    tok = _train_spe()
    ids = tok.encode("CCO")
    result = tok.decode(ids)
    assert isinstance(result, str)
    assert len(result) > 0


def test_train_only_fit():
    """SPE should not crash when fitting on tiny data."""
    tok = SPETokenizer(vocab_size=3, min_frequency=1, max_len=256)
    tok.fit(["CCO"])
    assert tok.vocab_size_real >= 4
    assert tok.merges is not None

    # Encode should work even with minimal training
    ids = tok.encode("CCO")
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID


def test_bos_eos_behavior():
    tok = _train_spe()
    ids = tok.encode("CCO")
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID


def test_empty_fit():
    """Fitting on an empty list should not crash."""
    tok = SPETokenizer(vocab_size=10, min_frequency=2, max_len=256)
    tok.fit([])
    # Should still be usable
    ids = tok.encode("CCO")
    assert len(ids) >= 2
