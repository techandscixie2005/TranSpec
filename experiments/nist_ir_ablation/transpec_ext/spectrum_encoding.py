import math

import torch
from torch import nn


class FourierWavenumberEncoding(nn.Module):
    """Fourier encoding of the wavenumber position axis.

    Only modifies spectral input embedding — never touches attention,
    masks, or decoder positional encoding.

    Input:  (B, 1, N) — intensity only
    Output: (B, 1+2*L, N) — intensity + sin/cos features

    Normalized wavenumber: p = (nu - nu_min) / (nu_max - nu_min)
    Features: sin(2pi * 2^f * p), cos(2pi * 2^f * p) for f = 0..L-1
    """

    def __init__(
        self,
        num_frequencies: int,
        nu_min: float,
        nu_max: float,
        n_points: int,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.n_points = n_points

        # Build normalized grid: (N,)
        grid = torch.linspace(nu_min, nu_max, n_points)
        p = (grid - nu_min) / (nu_max - nu_min)  # [0, 1]

        # Build features: (2*L, N)
        features = []
        for f in range(num_frequencies):
            omega = (2.0 ** f) * math.pi * 2.0
            features.append(torch.sin(omega * p))
            features.append(torch.cos(omega * p))
        basis = torch.stack(features, dim=0)  # (2*L, N)

        # Register as buffer: (1, 2*L, N) so expand(B, -1, -1) works
        self.register_buffer("fourier_basis", basis.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier wavenumber encoding.

        Args:
            x: (B, 1, N) intensity-only spectrum.

        Returns:
            (B, 1+2*L, N) intensity + sin/cos features.
        """
        B, C, N = x.shape
        if C != 1:
            raise ValueError(f"Expected 1 input channel, got {C}")
        if N != self.n_points:
            raise ValueError(f"Expected {self.n_points} points, got {N}")

        basis = self.fourier_basis.expand(B, -1, -1)  # (B, 2*L, N)
        return torch.cat([x, basis], dim=1)  # (B, 1+2*L, N)

    def output_channels(self) -> int:
        """Return the number of output channels."""
        return 1 + 2 * self.num_frequencies
