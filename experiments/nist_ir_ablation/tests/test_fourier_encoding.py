"""Tests for FourierWavenumberEncoding."""

import math

import pytest
import torch

from experiments.nist_ir_ablation.transpec_ext.spectrum_encoding import (
    FourierWavenumberEncoding,
)


class TestFourierWavenumberEncoding:
    """Tests for Fourier wavenumber encoding of spectra."""

    @pytest.fixture
    def encoding(self):
        return FourierWavenumberEncoding(
            num_frequencies=16,
            nu_min=552.0,
            nu_max=3844.0,
            n_points=3000,
        )

    def test_output_channels(self, encoding):
        """output_channels returns 1 + 2*L."""
        assert encoding.output_channels() == 1 + 2 * 16  # 33

    def test_forward_shape(self, encoding):
        """(B, 1, N) -> (B, 1+2L, N)."""
        x = torch.randn(4, 1, 3000)
        out = encoding(x)
        assert out.shape == (4, 33, 3000)

    def test_batch_expand(self, encoding):
        """Different batch sizes work."""
        for B in [1, 2, 8]:
            x = torch.randn(B, 1, 3000)
            out = encoding(x)
            assert out.shape == (B, 33, 3000)

    def test_fourier_basis_buffer(self, encoding):
        """fourier_basis is a registered buffer on correct device."""
        assert hasattr(encoding, "fourier_basis")
        assert encoding.fourier_basis.shape == (1, 32, 3000)

    def test_fourier_basis_values(self, encoding):
        """Sin features in [0,1) range, cos features similar."""
        basis = encoding.fourier_basis.squeeze(0)  # (32, 3000)
        sin_features = basis[0::2]  # even indices
        cos_features = basis[1::2]  # odd indices
        assert sin_features.shape == (16, 3000)
        assert cos_features.shape == (16, 3000)
        # Values should be bounded in [-1, 1]
        assert sin_features.abs().max().item() <= 1.0 + 1e-6
        assert cos_features.abs().max().item() <= 1.0 + 1e-6

    def test_fourier_sine_zero_at_start(self, encoding):
        """sin(0) = 0 for all frequencies at p=0 (nu_min)."""
        basis = encoding.fourier_basis.squeeze(0)  # (32, 3000)
        sin_features = basis[0::2]  # (16, 3000)
        # At p=0 (first point), sin(0) = 0 for all frequencies
        assert torch.allclose(sin_features[:, 0], torch.zeros(16), atol=1e-6)

    def test_fourier_cosine_one_at_start(self, encoding):
        """cos(0) = 1 for all frequencies at p=0 (nu_min)."""
        basis = encoding.fourier_basis.squeeze(0)  # (32, 3000)
        cos_features = basis[1::2]  # (16, 3000)
        assert torch.allclose(cos_features[:, 0], torch.ones(16), atol=1e-6)

    def test_input_channels_error(self, encoding):
        """Raises ValueError for wrong input channels."""
        x = torch.randn(4, 3, 3000)
        with pytest.raises(ValueError, match="Expected 1 input channel"):
            encoding(x)

    def test_n_points_error(self, encoding):
        """Raises ValueError for wrong number of points."""
        x = torch.randn(4, 1, 100)
        with pytest.raises(ValueError, match="Expected 3000 points"):
            encoding(x)

    def test_different_num_frequencies(self):
        """Different L produces (B, 1+2L, N)."""
        for L in [1, 8, 32, 64]:
            enc = FourierWavenumberEncoding(
                num_frequencies=L, nu_min=0.0, nu_max=1000.0, n_points=500
            )
            assert enc.output_channels() == 1 + 2 * L
            x = torch.randn(2, 1, 500)
            out = enc(x)
            assert out.shape == (2, 1 + 2 * L, 500)

    def test_identity_first_channel(self, encoding):
        """First channel (intensity) passes through unchanged."""
        x = torch.randn(4, 1, 3000)
        out = encoding(x)
        assert torch.allclose(out[:, 0:1, :], x)

    def test_different_nu_range(self):
        """Works with non-standard wavenumber ranges."""
        enc = FourierWavenumberEncoding(
            num_frequencies=8, nu_min=400.0, nu_max=4000.0, n_points=1000
        )
        x = torch.randn(1, 1, 1000)
        out = enc(x)
        assert out.shape == (1, 17, 1000)

    def test_omega_values(self, encoding):
        """Omega = 2*pi*2^f for f=0..L-1."""
        # Access internal parameters through the basis
        basis = encoding.fourier_basis.squeeze(0)  # (32, 3000)
        p = 0.5  # midway point

        # For f=0: sin(2*pi*1*0.5) = sin(pi) = 0
        # Midpoint index
        mid = 1500
        sin_f0 = basis[0, mid]
        expected_sin = math.sin(2.0 * math.pi * (2.0 ** 0) * 0.5)
        assert abs(sin_f0.item() - expected_sin) < 0.05

    def test_grad_flows(self, encoding):
        """Gradient flows through the encoding (copy op)."""
        x = torch.randn(2, 1, 3000, requires_grad=True)
        out = encoding(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        # Gradient should be 1.0 for first channel (copy)
        assert torch.allclose(x.grad, torch.ones_like(x.grad))
