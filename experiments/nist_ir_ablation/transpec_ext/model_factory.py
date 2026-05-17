import json
import os
import sys
from typing import Any, Dict, Optional

import torch
from torch import nn

# Add project root and src/ for legacy imports
# src/ is needed because src/model.py does: from util import PositionalEncoding
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for p in [_PROJECT_ROOT, _SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.model import Model

from .spectrum_encoding import FourierWavenumberEncoding
from .vocab import SPECIAL_TOKENS


class AblationModel(nn.Module):
    """Optional Fourier encoding -> core Model. Never modifies attention."""

    def __init__(
        self,
        core_model: Model,
        fourier_encoding: Optional[FourierWavenumberEncoding] = None,
    ):
        super().__init__()
        self.core = core_model
        self.fourier_encoding = fourier_encoding

    def forward(self, en, de_1, tgt_mask, tgt_key_padding_mask):
        if self.fourier_encoding is not None:
            # (B, 1, N) -> (B, 1+2L, N)
            en = self.fourier_encoding(en)
        return self.core(en, de_1, tgt_mask, tgt_key_padding_mask)

    @property
    def device(self):
        return next(self.core.parameters()).device


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def build_model(
    config: Dict[str, Any],
    vocab_size: int,
) -> AblationModel:
    """Build an AblationModel from config.

    Args:
        config: Full merged configuration dict.
        vocab_size: Size of tokenizer vocabulary (including special tokens).

    Returns:
        Configured AblationModel.
    """
    model_cfg = config.get("model", {})
    fourier_cfg = config.get("fourier", {})
    spectrum_cfg = config.get("spectrum", {})
    decoding_cfg = config.get("decoding", {})

    use_fourier = config.get("spectral_fourier_encoding", False)
    num_frequencies = fourier_cfg.get("num_frequencies", 32)
    n_points = spectrum_cfg.get("target_len", 3000)
    x_min = spectrum_cfg.get("x_min", 552.0)
    x_max = spectrum_cfg.get("x_max", 3844.0)

    # Determine input channels
    if use_fourier:
        input_channels = 1 + 2 * num_frequencies
    else:
        input_channels = 1

    d_model = model_cfg.get("d_model", 256)
    nhead = model_cfg.get("nhead", 8)
    num_encoder_layers = model_cfg.get("num_encoder_layers", 4)
    num_decoder_layers = model_cfg.get("num_decoder_layers", 4)
    dim_feedforward = model_cfg.get("dim_feedforward", 1024)
    dropout = model_cfg.get("dropout", 0.2)
    max_len = decoding_cfg.get("max_len", model_cfg.get("max_len", 256))
    use_cnn = model_cfg.get("use_cnn", True)
    use_mlp = model_cfg.get("use_mlp", False)

    core = Model(
        d_model=d_model,
        en_layers=num_encoder_layers,
        de_layers=num_decoder_layers,
        en_head=nhead,
        de_head=nhead,
        en_dim_feed=dim_feedforward,
        de_dim_feed=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        vocab_size=vocab_size,
        bias=True,
        use_cnn=use_cnn,
        use_mlp=use_mlp,
        input_channels=input_channels,
        reshape_size=10,
    )

    fourier_enc = None
    if use_fourier:
        fourier_enc = FourierWavenumberEncoding(
            num_frequencies=num_frequencies,
            nu_min=x_min,
            nu_max=x_max,
            n_points=n_points,
        )

    model = AblationModel(core_model=core, fourier_encoding=fourier_enc)
    return model


def save_model_summary(model: AblationModel, path: str) -> None:
    """Save model architecture summary with parameter counts."""
    params = count_parameters(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines = [
        "=" * 60,
        "Model Summary",
        "=" * 60,
        f"Total params:     {params['total_params']:,}",
        f"Trainable params: {params['trainable_params']:,}",
        "",
        "Architecture:",
        f"  Fourier encoding: {'yes' if model.fourier_encoding is not None else 'no'}",
        f"  Core encoder layers: {model.core.encoder_layer.num_layers if hasattr(model.core, 'encoder_layer') else 'N/A'}",
        "",
    ]

    try:
        lines.append(f"  Input channels:   {model.core.input_channels}")
    except AttributeError:
        pass

    with open(path, "w") as f:
        f.write("\n".join(lines))
