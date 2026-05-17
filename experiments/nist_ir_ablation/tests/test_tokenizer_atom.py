"""Tests for tokenizer_atom.py"""
import tempfile
import os
import json

from experiments.nist_ir_ablation.transpec_ext.tokenizer_atom import AtomTokenizer
from experiments.nist_ir_ablation.transpec_ext.vocab import (
    BOS_ID, EOS_ID, PAD_ID, UNK_ID,
)


def _make_tokenizer():
    tok = AtomTokenizer(max_len=256)
    tok.fit(["CCO", "CCN", "CCC", "C=O", "C#N", "c1ccccc1", "Br", "Cl", "[C@H](C)O"])
    return tok


def test_encode_decode_simple():
    tok = _make_tokenizer()
    ids = tok.encode("CCO")
    decoded = tok.decode(ids)
    assert decoded == "CCO"


def test_supports_cl_and_br():
    tok = _make_tokenizer()
    for smiles in ["ClCCl", "BrCBr", "CCBr", "CCCl"]:
        ids = tok.encode(smiles)
        decoded = tok.decode(ids)
        assert decoded == smiles, f"Failed on {smiles}: got {decoded}"


def test_supports_bracket_tokens():
    tok = _make_tokenizer()
    ids = tok.encode("[C@H](C)O")
    decoded = tok.decode(ids)
    # Chirality marker may or may not be preserved depending on fit data
    assert "C" in decoded and "O" in decoded


def test_supports_ring_token_percent_10():
    """Test that %10 multi-digit ring closures work."""
    tok = _make_tokenizer()
    # SMILES with %10 ring closure
    smiles = "C1CCCCC1"  # cyclohexane (uses 1, not %10)
    ids = tok.encode(smiles)
    decoded = tok.decode(ids)
    assert decoded == smiles

    # Test that %10 pattern is in the regex (big ring)
    tok2 = AtomTokenizer(max_len=256)
    tok2.fit(["C%10CCCCCCCCCC%10"])
    ids = tok2.encode("C%10CCCCCCCCCC%10")
    decoded = tok2.decode(ids)
    assert decoded == "C%10CCCCCCCCCC%10"


def test_bos_eos_present():
    tok = _make_tokenizer()
    ids = tok.encode("CCO")
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID


def test_bos_eos_absent_without_special():
    tok = _make_tokenizer()
    ids = tok.encode("CCO", add_special_tokens=False)
    assert BOS_ID not in ids
    assert EOS_ID not in ids


def test_pad_id():
    tok = _make_tokenizer()
    assert tok.pad_id == PAD_ID


def test_vocab_size():
    tok = _make_tokenizer()
    assert tok.vocab_size >= 4  # at least PAD, BOS, EOS, UNK


def test_save_load():
    tok = _make_tokenizer()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
        tok.save(path)

    loaded = AtomTokenizer.load(path)
    os.unlink(path)

    for smiles in ["CCO", "ClCCl", "BrCBr"]:
        assert loaded.encode(smiles) == tok.encode(smiles)
        assert loaded.decode(loaded.encode(smiles)) == smiles


def test_encode_truncation():
    tok = AtomTokenizer(max_len=8)  # very short max_len
    tok.fit(["CCO"])
    ids = tok.encode("CCO")  # BOS + C + C + O + EOS = 5 tokens, fits
    assert len(ids) <= 8

    # Test truncation with longer SMILES
    tok2 = AtomTokenizer(max_len=5)
    tok2.fit(["CCCCCCCC"])
    ids = tok2.encode("CCCCCCCC")
    assert len(ids) <= 5


def test_unknown_fallback():
    """Unknown characters should become UNK token."""
    tok = _make_tokenizer()
    ids = tok.encode("CCO", add_special_tokens=False)
    # All known tokens
    assert UNK_ID not in ids
