from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


def resample_spectrum(
    wavenumbers,
    intensities,
    target_low: float = 552.0,
    target_high: float = 3844.0,
    n_points: int = 3000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample an IR spectrum to a fixed grid.

    Args:
        wavenumbers: Array of wavenumber values.
        intensities: Array of intensity values.
        target_low: Lower bound of the target grid.
        target_high: Upper bound of the target grid.
        n_points: Number of points in the target grid.

    Returns:
        (grid, spectrum) where grid is the target wavenumber grid
        and spectrum is the interpolated intensity values.
        Out-of-range regions are filled with 0.0.
    """
    x = np.asarray(wavenumbers, dtype=float)
    y = np.asarray(intensities, dtype=float)

    # Sort by wavenumber ascending
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Merge duplicate x values by averaging y
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) < len(x):
        y_merged = np.zeros_like(unique_x, dtype=float)
        count = np.zeros_like(unique_x, dtype=float)
        np.add.at(y_merged, inverse, y)
        np.add.at(count, inverse, 1.0)
        y = y_merged / count
        x = unique_x

    grid = np.linspace(target_low, target_high, n_points)

    # Find the data range
    in_range = (grid >= x.min()) & (grid <= x.max())

    if in_range.sum() == 0:
        # No overlap at all
        return grid, np.zeros(n_points, dtype=float)

    # Interpolate within the data range
    if len(x) < 4:
        kind = "linear"
    else:
        kind = "cubic"

    f_interp = interp1d(x, y, kind=kind, bounds_error=False, fill_value=0.0)
    spectrum = f_interp(grid)

    # Ensure out-of-range is exactly 0.0 (interp1d can produce NaN at boundaries)
    spectrum[~in_range] = 0.0

    return grid, spectrum


def normalize_spectrum(spectrum: np.ndarray, mode: str = "max") -> np.ndarray:
    """
    Normalize a spectrum array.

    Args:
        spectrum: 1D numpy array of intensity values.
        mode: Normalization mode. "max" divides by max positive intensity (zero-safe).

    Returns:
        Normalized spectrum.
    """
    if mode == "max":
        max_val = spectrum.max()
        if max_val > 0:
            return spectrum / max_val
        return spectrum
    elif mode == "max_100":
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0, 100))
        return scaler.fit_transform(spectrum.reshape(-1, 1)).flatten()
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def validate_spectrum(spectrum: np.ndarray, expected_len: int = 3000) -> Tuple[bool, str]:
    """Validate a processed spectrum. Returns (is_valid, reason)."""
    if spectrum.ndim != 1:
        return False, f"Expected 1D array, got {spectrum.ndim}D"
    if len(spectrum) != expected_len:
        return False, f"Expected length {expected_len}, got {len(spectrum)}"
    if np.any(np.isnan(spectrum)):
        return False, "Contains NaN values"
    if np.any(np.isinf(spectrum)):
        return False, "Contains inf values"
    return True, "ok"


def preprocess_record(
    wavenumbers,
    intensities,
    n_points: int = 3000,
    low: float = 552.0,
    high: float = 3844.0,
    normalize: str = "max",
    clip_negative: bool = True,
    filter_all_zero: bool = False,
) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline for a single record: resample -> clip -> normalize -> validate.

    Returns a 1D numpy array of length n_points, or None if validation fails.
    """
    _, spectrum = resample_spectrum(
        wavenumbers, intensities,
        target_low=low, target_high=high, n_points=n_points,
    )

    if clip_negative:
        spectrum = np.clip(spectrum, 0.0, None)

    spectrum = normalize_spectrum(spectrum, mode=normalize)

    is_valid, reason = validate_spectrum(spectrum, expected_len=n_points)
    if not is_valid:
        return None

    if filter_all_zero and spectrum.max() == 0.0:
        return None

    return spectrum
