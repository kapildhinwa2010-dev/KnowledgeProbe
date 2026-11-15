# KnowledgeProbe - Probe spectral behavior signatures of activation sets
# Part of the Kapildhinwa2010-dev Toolkit.

"""BehaviorNet - model behavior signatures and power-law spectra.

Lineage: Reverse.Dean (behave_yourself.py). A model's "behavior" is captured
by the spectrum of its activation/weight matrices. Real trained networks show
power-law decay; this module fits a power law, produces a SHA-256 behavior
signature and compares two models by spectral distance.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = [
    "singular_spectrum",
    "fit_power_law",
    "behavior_signature",
    "spectral_distance",
    "is_power_law",
]


def singular_spectrum(matrix: np.ndarray) -> np.ndarray:
    """Singular values (descending) of a 2-D matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    return np.linalg.svd(matrix, compute_uv=False)


def fit_power_law(spectrum: np.ndarray):
    """Fit log(sigma) = a * log(rank) + b; returns (slope, intercept, r_squared)."""
    s = np.asarray(spectrum, dtype=np.float64)
    if s.size < 3:
        raise ValueError("spectrum needs at least 3 values")
    s = s[s > 0]
    if s.size < 3:
        raise ValueError("spectrum needs at least 3 positive values")
    x = np.log(np.arange(1, s.size + 1, dtype=np.float64))
    y = np.log(s)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), r_squared


def behavior_signature(matrix: np.ndarray, bins: int = 16) -> str:
    """Deterministic SHA-256 signature of a matrix's log-spectrum histogram."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    s = singular_spectrum(matrix)
    s = s[s > 0]
    if s.size == 0:
        s = np.array([1e-30])
    hist, _ = np.histogram(np.log10(s), bins=bins)
    payload = hist.astype(np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def spectral_distance(matrix_a: np.ndarray, matrix_b: np.ndarray, k: int = 8) -> float:
    """Relative L2 distance between the top-k log-singular-values."""
    if k < 1:
        raise ValueError("k must be >= 1")
    sa = singular_spectrum(matrix_a)[:k]
    sb = singular_spectrum(matrix_b)[:k]
    la = np.log10(np.maximum(sa, 1e-30))
    lb = np.log10(np.maximum(sb, 1e-30))
    return float(np.linalg.norm(la - lb) / np.sqrt(k))


def is_power_law(spectrum: np.ndarray, min_r_squared: float = 0.9) -> bool:
    """True when the spectrum follows a power law well enough."""
    if not 0.0 <= min_r_squared <= 1.0:
        raise ValueError("min_r_squared must be in [0, 1]")
    _, _, r_squared = fit_power_law(spectrum)
    return bool(r_squared >= min_r_squared)
