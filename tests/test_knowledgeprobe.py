import numpy as np
import pytest

from knowledgeprobe import (
    behavior_signature,
    fit_power_law,
    is_power_law,
    singular_spectrum,
    spectral_distance,
)


def _power_law_matrix(rows=64, cols=64, exponent=1.5, seed=0):
    rng = np.random.default_rng(seed)
    ranks = min(rows, cols)
    s = 1.0 / np.arange(1, ranks + 1, dtype=np.float64) ** exponent
    u, _ = np.linalg.qr(rng.standard_normal((rows, ranks)))
    vt, _ = np.linalg.qr(rng.standard_normal((ranks, cols)))
    return (u * s) @ vt


def test_spectrum_descending():
    s = singular_spectrum(_power_law_matrix())
    assert np.all(np.diff(s) <= 1e-12)


def test_spectrum_rejects_1d():
    with pytest.raises(ValueError):
        singular_spectrum(np.ones(5))


def test_power_law_fit_recovers_exponent():
    W = _power_law_matrix(exponent=1.5)
    s = singular_spectrum(W)
    slope, _, r2 = fit_power_law(s)
    assert slope == pytest.approx(-1.5, abs=0.3)
    assert r2 > 0.9


def test_power_law_fit_rejects_short():
    with pytest.raises(ValueError):
        fit_power_law(np.array([1.0, 0.5]))


def test_signature_deterministic():
    W = _power_law_matrix()
    assert behavior_signature(W) == behavior_signature(W)


def test_signature_changes_with_matrix():
    # Same exponent => identical singular values regardless of seed, so signatures
    # legitimately match. Use different exponents to get different spectra.
    w1 = _power_law_matrix(exponent=1.5, seed=1)
    w2 = _power_law_matrix(exponent=2.5, seed=1)
    assert behavior_signature(w1) != behavior_signature(w2)


def test_signature_is_hex():
    sig = behavior_signature(np.eye(10))
    assert len(sig) == 64
    int(sig, 16)


def test_spectral_distance_zero_for_identical():
    W = _power_law_matrix()
    assert spectral_distance(W, W) == pytest.approx(0.0)


def test_spectral_distance_positive():
    w1 = _power_law_matrix(exponent=1.0)
    w2 = _power_law_matrix(exponent=2.0)
    assert spectral_distance(w1, w2) > 0.0


def test_is_power_law_true_for_constructed():
    W = _power_law_matrix()
    assert is_power_law(singular_spectrum(W))


def test_is_power_law_false_for_flat():
    s = np.ones(16)
    assert not is_power_law(s)


def test_is_power_law_rejects_bad_threshold():
    with pytest.raises(ValueError):
        is_power_law(np.ones(8), min_r_squared=2.0)
