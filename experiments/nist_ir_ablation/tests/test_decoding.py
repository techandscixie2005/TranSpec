"""Tests for decoding.py: threshold-value search logic.

Uses a mock core model with deterministic outputs so tests verify
the decoding algorithm rather than model accuracy.
"""

import pytest
import torch
from torch import nn

from experiments.nist_ir_ablation.transpec_ext.decoding import (
    decode_ablation,
    threshold_value_search_ablation,
)
from experiments.nist_ir_ablation.transpec_ext.model_factory import AblationModel
from experiments.nist_ir_ablation.transpec_ext.vocab import (
    BOS_ID, EOS_ID, PAD_ID, FIRST_REAL_TOKEN_ID,
)


# ── helpers ────────────────────────────────────────────────────────────


class MockPreType(nn.Module):
    """Outputs logits where a configurable token ID has highest probability.

    After ``n_greedy`` steps, switches to predicting EOS_ID so the
    decoding terminates.
    """

    def __init__(self, vocab_size: int, n_greedy: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_greedy = n_greedy
        self.step_counter = 0
        self.token_seq = []

    def forward(self, x):
        B, S, D = x.shape
        logits = torch.full((B, S, self.vocab_size), -10.0)

        # The last position in the sequence is the one we're predicting
        # Always predict EOS after n_greedy steps (counting this step)
        self.step_counter += 1
        if self.step_counter > self.n_greedy:
            token_id = EOS_ID  # emit EOS
        else:
            token_id = FIRST_REAL_TOKEN_ID + 1  # e.g. "C" for atom tokenizer

        self.token_seq.append(token_id)
        logits[:, -1, token_id] = 0.0  # highest logit
        return logits

    def reset(self):
        self.step_counter = 0
        self.token_seq = []


class MockDecoderLayer(nn.Module):
    """Returns ``tgt`` unchanged (identity for decoder)."""

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        return tgt


def _make_core_model(
    vocab_size: int,
    d_model: int = 16,
    n_greedy: int = 3,
) -> nn.Module:
    """Build a mock core model that produces deterministic outputs."""
    pre_type = MockPreType(vocab_size, n_greedy=n_greedy)

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Identity()  # bypass CNN
            self.encoder = nn.Identity()  # bypass encoder
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pe = nn.Identity()  # bypass PE
            self.decoder = MockDecoderLayer()
            self.pre_type = pre_type

        def forward(self, en, de_1, tgt_mask, tgt_key_padding_mask):
            x = self.encoder(en)
            tgt = self.embedding(de_1)
            tgt = self.pe(tgt)
            tgt = self.decoder(
                tgt, x,
                tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            )
            return self.pre_type(tgt)

    return Core(), pre_type


class MockTokenizer:
    """Minimal tokenizer that maps token IDs to dummy SMILES strings."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._id_to_token = {i: f"T{i}" for i in range(vocab_size)}

    def decode(self, token_ids, skip_special_tokens=True, stop_at_eos=True):
        tokens = []
        for tid in token_ids:
            if stop_at_eos and tid == EOS_ID:
                break
            if skip_special_tokens and tid in (PAD_ID, BOS_ID, EOS_ID):
                continue
            tokens.append(self._id_to_token.get(tid, "?"))
        return "".join(tokens)


@pytest.fixture
def model_and_tokenizer():
    vocab_size = 12
    core, pre_type = _make_core_model(vocab_size, d_model=8, n_greedy=3)
    model = AblationModel(core_model=core, fourier_encoding=None)
    tokenizer = MockTokenizer(vocab_size)
    return model, tokenizer, pre_type


# Need pytest
import pytest


# ── tests ──────────────────────────────────────────────────────────────


def test_decoder_stops_on_eos(model_and_tokenizer):
    """Decoding terminates when EOS token is emitted."""
    model, tokenizer, pre_type = model_and_tokenizer
    pre_type.reset()
    spectrum = torch.randn(1, 1, 3000)

    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        max_len=256, threshold_value=0.01, candidate_limit=10,
    )

    # Should have produced some candidates
    assert len(candidates) > 0
    # Token sequences should not contain EOS in the middle (decode strips it)
    for token_ids in ids:
        if token_ids:
            assert EOS_ID not in token_ids, "EOS should be stripped from decoded IDs"


def test_decoder_no_bos_in_output(model_and_tokenizer):
    """Decoded SMILES and token IDs must not contain BOS/PAD."""
    model, tokenizer, pre_type = model_and_tokenizer
    pre_type.reset()
    spectrum = torch.randn(1, 1, 3000)

    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        max_len=256, threshold_value=0.01, candidate_limit=10,
    )

    for token_ids in ids:
        assert BOS_ID not in token_ids
        assert PAD_ID not in token_ids


def test_returns_ranked_candidates(model_and_tokenizer):
    """Candidates are returned in descending order of model score."""
    model, tokenizer, pre_type = model_and_tokenizer
    pre_type.reset()
    spectrum = torch.randn(1, 1, 3000)

    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        max_len=256, threshold_value=0.01, candidate_limit=10,
    )

    # Scores should be descending
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], "Candidates not sorted by score descending"


def test_handles_empty_candidates_gracefully(model_and_tokenizer):
    """If model fails to produce output, empty lists are returned without crash."""
    model, tokenizer, pre_type = model_and_tokenizer
    pre_type.reset()

    # Build a custom model with pre_type that always outputs -inf
    class BrokenPreType(nn.Module):
        def forward(self_, x):
            B, S, D = x.shape
            return torch.full((B, S, tokenizer.vocab_size), -float("inf"))

    core, _ = _make_core_model(tokenizer.vocab_size)
    core.pre_type = BrokenPreType()
    broken_model = AblationModel(core_model=core, fourier_encoding=None)

    spectrum = torch.randn(1, 1, 3000)
    candidates, scores, ids = decode_ablation(
        broken_model, spectrum, tokenizer,
        max_len=64, threshold_value=0.01, candidate_limit=10,
    )
    # Should not crash; may return empty or fallback results
    assert isinstance(candidates, list)
    assert isinstance(scores, list)
    assert isinstance(ids, list)


def test_respects_max_decode_len():
    """Short max_len forces early termination before EOS is predicted."""
    vocab_size = 12
    core, pre_type = _make_core_model(vocab_size, d_model=8, n_greedy=100)
    model = AblationModel(core_model=core, fourier_encoding=None)
    tokenizer = MockTokenizer(vocab_size)

    pre_type.reset()
    spectrum = torch.randn(1, 1, 3000)

    # Very short max_len = 2 means only 1 real token before forced stop
    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        max_len=2, threshold_value=0.01, candidate_limit=10,
    )

    # Should complete without crash; token sequences at most max_len
    for token_ids in ids:
        assert len(token_ids) <= 2


def test_does_not_crash_on_cpu():
    """Decoding works on CPU."""
    vocab_size = 12
    core, pre_type = _make_core_model(vocab_size, d_model=8, n_greedy=2)
    model = AblationModel(core_model=core, fourier_encoding=None)
    tokenizer = MockTokenizer(vocab_size)

    pre_type.reset()
    spectrum = torch.randn(1, 1, 3000)

    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        max_len=64, threshold_value=0.01, candidate_limit=10,
    )
    assert len(candidates) >= 0


def test_threshold_value_controls_branching():
    """Lower threshold should create more branches (candidates)."""
    vocab_size = 12
    core, pre_type = _make_core_model(vocab_size, d_model=8, n_greedy=3)
    model = AblationModel(core_model=core, fourier_encoding=None)
    tokenizer = MockTokenizer(vocab_size)

    pre_type.reset()
    spectrum = torch.randn(1, 1, 3000)

    candidates_high, _, _ = decode_ablation(
        model, spectrum, tokenizer,
        max_len=64, threshold_value=0.5, candidate_limit=100,
    )
    pre_type.reset()
    candidates_low, _, _ = decode_ablation(
        model, spectrum, tokenizer,
        max_len=64, threshold_value=0.001, candidate_limit=100,
    )

    # Lower threshold should give at least as many candidates
    assert len(candidates_low) >= len(candidates_high)


# ── Beam search tests ────────────────────────────────────────────────


class BeamMockPreType(nn.Module):
    """Produces a multi-token distribution so beam search can branch.

    At each step before ``eos_at_step``, tokens 4/5/6 and EOS have
    distinguishable scores.  At or after ``eos_at_step``, only EOS
    scores highly, terminating all active hypotheses.
    """

    def __init__(self, vocab_size: int, eos_at_step: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_at_step = eos_at_step

    def forward(self, x):
        B, S, D = x.shape
        logits = torch.full((B, S, self.vocab_size), -100.0)

        if S >= self.eos_at_step:
            logits[:, -1, EOS_ID] = 0.0
        else:
            logits[:, -1, FIRST_REAL_TOKEN_ID] = 0.0     # token 4
            logits[:, -1, FIRST_REAL_TOKEN_ID + 1] = -1.0  # token 5
            logits[:, -1, FIRST_REAL_TOKEN_ID + 2] = -2.0  # token 6
            logits[:, -1, EOS_ID] = -0.5
        return logits


def _make_beam_core(
    vocab_size: int,
    d_model: int = 16,
    eos_at_step: int = 4,
) -> nn.Module:
    """Build a mock core model with BeamMockPreType."""
    pre_type = BeamMockPreType(vocab_size, eos_at_step=eos_at_step)

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Identity()
            self.encoder = nn.Identity()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pe = nn.Identity()
            self.decoder = MockDecoderLayer()
            self.pre_type = pre_type

    return Core()


@pytest.fixture
def beam_model_and_tokenizer():
    vocab_size = 12
    core = _make_beam_core(vocab_size, d_model=8, eos_at_step=4)
    model = AblationModel(core_model=core, fourier_encoding=None)
    tokenizer = MockTokenizer(vocab_size)
    return model, tokenizer


def test_beam_returns_ranked_candidates(beam_model_and_tokenizer):
    """Beam decoder returns candidates in descending score order."""
    model, tokenizer = beam_model_and_tokenizer
    spectrum = torch.randn(1, 1, 3000)

    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        decode_method="beam", beam_size=5, candidate_limit=10, max_len=256,
    )

    assert len(candidates) > 0
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], "Candidates not sorted by score descending"


def test_beam_respects_beam_size(beam_model_and_tokenizer):
    """Number of candidates respects candidate_limit (beam explores diversity)."""
    model, tokenizer = beam_model_and_tokenizer
    spectrum = torch.randn(1, 1, 3000)

    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        decode_method="beam", beam_size=3, candidate_limit=20, max_len=256,
    )

    assert 1 <= len(candidates) <= 20


def test_beam_respects_max_decode_len():
    """Short max_len forces early termination before EOS."""
    vocab_size = 12
    core = _make_beam_core(vocab_size, d_model=8, eos_at_step=100)  # never EOS
    model = AblationModel(core_model=core, fourier_encoding=None)
    tokenizer = MockTokenizer(vocab_size)

    spectrum = torch.randn(1, 1, 3000)
    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        decode_method="beam", beam_size=3, candidate_limit=10, max_len=3,
    )

    # Token sequences should be at most max_len + 1 (BOS token)
    for token_ids in ids:
        assert len(token_ids) <= 3


def test_beam_stops_on_eos(beam_model_and_tokenizer):
    """Beam decoder terminates hypotheses when EOS is generated."""
    model, tokenizer = beam_model_and_tokenizer
    spectrum = torch.randn(1, 1, 3000)

    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        decode_method="beam", beam_size=5, candidate_limit=10, max_len=256,
    )

    assert len(candidates) > 0
    for token_ids in ids:
        assert EOS_ID not in token_ids, "EOS should be stripped from decoded IDs"


def test_beam_excludes_special_tokens(beam_model_and_tokenizer):
    """BOS, EOS, PAD must not appear in decoded SMILES or token IDs."""
    model, tokenizer = beam_model_and_tokenizer
    spectrum = torch.randn(1, 1, 3000)

    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        decode_method="beam", beam_size=5, candidate_limit=10, max_len=256,
    )

    for smi in candidates:
        assert "<BOS>" not in smi
        assert "<EOS>" not in smi
        assert "<PAD>" not in smi

    for token_ids in ids:
        assert BOS_ID not in token_ids
        assert EOS_ID not in token_ids
        assert PAD_ID not in token_ids


def test_beam_fewer_than_limit(beam_model_and_tokenizer):
    """Beam decoder handles cases where fewer than candidate_limit are produced."""
    model, tokenizer = beam_model_and_tokenizer
    spectrum = torch.randn(1, 1, 3000)

    # Short max_len limits total candidates below candidate_limit
    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        decode_method="beam", beam_size=2, candidate_limit=100, max_len=5,
    )

    assert len(candidates) < 100
    assert len(candidates) > 0
    assert len(scores) == len(candidates)
    assert len(ids) == len(candidates)


def test_beam_on_cpu(beam_model_and_tokenizer):
    """Beam decoder works on CPU."""
    model, tokenizer = beam_model_and_tokenizer
    spectrum = torch.randn(1, 1, 3000)

    candidates, scores, ids = decode_ablation(
        model, spectrum, tokenizer,
        decode_method="beam", beam_size=3, candidate_limit=10, max_len=64,
    )

    assert len(candidates) >= 0
