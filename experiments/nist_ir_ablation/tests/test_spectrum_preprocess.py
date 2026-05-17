"""Tests for spectrum_preprocess.py"""
import numpy as np

from experiments.nist_ir_ablation.transpec_ext.spectrum_preprocess import (
    resample_spectrum,
    normalize_spectrum,
    validate_spectrum,
    preprocess_record,
)


def test_output_shape_is_3000():
    x = np.linspace(552, 3844, 2500)
    y = np.random.rand(2500) * 100
    grid, spectrum = resample_spectrum(x, y)
    assert len(spectrum) == 3000
    assert len(grid) == 3000


def test_no_nan():
    x = np.linspace(552, 3844, 2500)
    y = np.random.rand(2500) * 100
    _, spectrum = resample_spectrum(x, y)
    assert not np.any(np.isnan(spectrum))
    assert not np.any(np.isinf(spectrum))


def test_negative_clipping():
    x = np.linspace(552, 3844, 2500)
    y = np.random.randn(2500) * 10  # has negative values
    result = preprocess_record(x, y, clip_negative=True)
    assert result is not None
    assert np.all(result >= 0.0)


def test_max_normalization():
    x = np.linspace(552, 3844, 2500)
    y = np.ones(2500) * 50.0
    # Add a peak
    y[1000] = 100.0
    result = preprocess_record(x, y, normalize="max")
    assert result is not None
    assert abs(result.max() - 1.0) < 1e-5


def test_out_of_range_fill():
    """Points outside the data range should be filled with 0."""
    x = np.linspace(1000, 3000, 500)  # only covers part of the range
    y = np.ones(500) * 10.0
    grid, spectrum = resample_spectrum(x, y, target_low=552, target_high=3844, n_points=100)
    # The first few points (at 552-1000) should be 0
    assert spectrum[0] == 0.0
    # The last few points (at 3000-3844) should be 0
    assert spectrum[-1] == 0.0
    # The middle (in-range) should be non-zero
    mid_indices = (grid >= 1000) & (grid <= 3000)
    assert spectrum[mid_indices].sum() > 0


def test_validate_spectrum():
    valid = np.zeros(3000)
    assert validate_spectrum(valid)[0] is True

    wrong_shape = np.zeros((3000, 1))
    assert validate_spectrum(wrong_shape)[0] is False

    with_nan = np.zeros(3000)
    with_nan[0] = np.nan
    assert validate_spectrum(with_nan)[0] is False


def test_normalize_zero_safe():
    """Normalizing an all-zero spectrum should not crash."""
    zeros = np.zeros(3000)
    result = normalize_spectrum(zeros, mode="max")
    assert np.all(result == 0.0)


def test_preprocess_record_invalid_returns_none():
    """A spectrum with all NaN should return None."""
    x = np.linspace(552, 3844, 100)
    y = np.full(100, np.nan)
    result = preprocess_record(x, y)
    assert result is None


def test_cubic_interpolation_fallback_to_linear():
    """With <4 points, interpolation should fall back to linear without error."""
    x = [552.0, 2000.0, 3844.0]
    y = [0.0, 1.0, 0.5]
    grid, spectrum = resample_spectrum(x, y, n_points=100)
    assert len(spectrum) == 100
    assert not np.any(np.isnan(spectrum))
