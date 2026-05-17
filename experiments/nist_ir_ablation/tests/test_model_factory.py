"""Tests for model_factory: build_model, AblationModel, count_parameters."""

import pytest
import torch

from experiments.nist_ir_ablation.transpec_ext.model_factory import (
    AblationModel,
    build_model,
    count_parameters,
)
from experiments.nist_ir_ablation.transpec_ext.spectrum_encoding import (
    FourierWavenumberEncoding,
)


def _minimal_config(use_fourier=False, vocab_size=38):
    """Create a minimal config dict for smoke-test model building."""
    return {
        "spectral_fourier_encoding": use_fourier,
        "model": {
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "use_cnn": True,
            "use_mlp": False,
            "max_len": 256,
        },
        "decoding": {"max_len": 256},
        "spectrum": {"target_len": 3000, "x_min": 552.0, "x_max": 3844.0},
        "fourier": {"num_frequencies": 16},
    }


class TestBuildModel:
    """Tests for build_model function."""

    def test_build_no_fourier(self):
        """Build model without Fourier encoding."""
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        assert isinstance(model, AblationModel)
        assert model.fourier_encoding is None
        assert model.core.input_channels == 1

    def test_build_with_fourier(self):
        """Build model with Fourier encoding."""
        config = _minimal_config(use_fourier=True)
        model = build_model(config, vocab_size=38)
        assert isinstance(model, AblationModel)
        assert model.fourier_encoding is not None
        assert isinstance(model.fourier_encoding, FourierWavenumberEncoding)
        # input_channels = 1 + 2*16 = 33
        assert model.core.input_channels == 33

    def test_build_vocab_size(self):
        """Vocab_size is reflected in output projection."""
        for vocab_size in [38, 50, 100]:
            config = _minimal_config(use_fourier=False)
            model = build_model(config, vocab_size=vocab_size)
            # Check output projection dimension
            assert model.core.pre_type[-1].out_features == vocab_size

    def test_forward_no_fourier_shape(self):
        """Forward pass produces (B, seq_len, vocab_size)."""
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        B, S = 4, 10
        en = torch.randn(B, 1, 3000)
        de_1 = torch.randint(0, 38, (B, S))
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(S)
        tgt_padding_mask = de_1 == 0

        out = model(en, de_1, tgt_mask, tgt_padding_mask)
        assert out.shape == (B, S, 38)

    def test_forward_with_fourier_shape(self):
        """Forward with Fourier produces same output shape."""
        config = _minimal_config(use_fourier=True)
        model = build_model(config, vocab_size=38)
        B, S = 4, 10
        en = torch.randn(B, 1, 3000)
        de_1 = torch.randint(0, 38, (B, S))
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(S)
        tgt_padding_mask = de_1 == 0

        out = model(en, de_1, tgt_mask, tgt_padding_mask)
        assert out.shape == (B, S, 38)

    def test_forward_1d_spectrum(self):
        """Spectrum with shape (1, 3000) is handled via AblationModel wrapper."""
        # build_model returns AblationModel which accepts (B, C, N)
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        B, S = 1, 5
        en = torch.randn(B, 1, 3000)
        de_1 = torch.randint(0, 38, (B, S))
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(S)
        tgt_padding_mask = de_1 == 0
        out = model(en, de_1, tgt_mask, tgt_padding_mask)
        assert out.shape == (B, S, 38)

    def test_gradient_flows(self):
        """Gradient flows through the full model."""
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        B, S = 2, 8
        en = torch.randn(B, 1, 3000)
        de_1 = torch.randint(0, 38, (B, S))
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(S)
        tgt_padding_mask = de_1 == 0

        out = model(en, de_1, tgt_mask, tgt_padding_mask)
        loss = out.sum()
        loss.backward()

        # Check that some gradients are non-zero
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowing through model"

    def test_multiple_batch_sizes(self):
        """Works with different batch sizes."""
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        for B in [1, 2, 8]:
            en = torch.randn(B, 1, 3000)
            de_1 = torch.randint(0, 38, (B, 5))
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(5)
            tgt_padding_mask = de_1 == 0
            out = model(en, de_1, tgt_mask, tgt_padding_mask)
            assert out.shape == (B, 5, 38)


class TestAblationModel:
    """Tests for the AblationModel wrapper."""

    def test_device_property(self):
        """device property returns the model's device."""
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        device = model.device
        assert device == next(model.core.parameters()).device

    def test_fourier_applied_in_forward(self):
        """Fourier encoding is applied before forward to core."""
        config = _minimal_config(use_fourier=True)
        model = build_model(config, vocab_size=38)
        B, S = 2, 5
        en = torch.randn(B, 1, 3000)
        de_1 = torch.randint(0, 38, (B, S))
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(S)
        tgt_padding_mask = de_1 == 0

        out = model(en, de_1, tgt_mask, tgt_padding_mask)
        assert out.shape == (B, S, 38), "Fourier-augmented forward failed"

    def test_no_fourier_passthrough(self):
        """Without Fourier, input passes directly to core."""
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        assert model.fourier_encoding is None


class TestCountParameters:
    """Tests for count_parameters."""

    def test_returns_dict(self):
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        counts = count_parameters(model)
        assert isinstance(counts, dict)
        assert "total_params" in counts
        assert "trainable_params" in counts

    def test_total_positive(self):
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        counts = count_parameters(model)
        assert counts["total_params"] > 0

    def test_trainable_equals_total(self):
        """All params are trainable by default."""
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        counts = count_parameters(model)
        assert counts["total_params"] == counts["trainable_params"]

    def test_fourier_adds_params(self):
        """Fourier model has more params (CNN input layer)."""
        config_no_f = _minimal_config(use_fourier=False)
        config_f = _minimal_config(use_fourier=True)
        model_no_f = build_model(config_no_f, vocab_size=38)
        model_f = build_model(config_f, vocab_size=38)
        counts_no_f = count_parameters(model_no_f)
        counts_f = count_parameters(model_f)
        # Fourier model should have more params (CNN input layer)
        assert counts_f["total_params"] > counts_no_f["total_params"]

    def test_no_negative_params(self):
        """No layer can have negative parameter count."""
        config = _minimal_config(use_fourier=False)
        model = build_model(config, vocab_size=38)
        counts = count_parameters(model)
        assert counts["total_params"] >= 0
        assert counts["trainable_params"] >= 0
