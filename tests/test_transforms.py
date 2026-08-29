"""Unit tests for src/transforms.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.transforms import (
    LOG_RANGE,
    LOG_SHIFT,
    RV_SCALE,
    build_har_features,
    clip_forecasts,
    compute_clip_threshold,
    proietti_lutkepohl_bias_correction,
    scale_log_rv,
    scale_rv,
    unscale_log_rv,
    unscale_rv,
)


# ---------------------------------------------------------------------------
# scale_rv / unscale_rv
# ---------------------------------------------------------------------------

def test_scale_rv_known_value():
    """scale_rv(1e-3) should equal exactly 1.0."""
    result = scale_rv(np.array([1e-3]))
    assert result[0] == 1.0, f"Expected 1.0, got {result[0]}"


def test_scale_rv_zero():
    """scale_rv(0) should equal 0."""
    result = scale_rv(np.array([0.0]))
    assert result[0] == 0.0, f"Expected 0.0, got {result[0]}"


def test_unscale_rv_roundtrip():
    """unscale_rv(scale_rv(x)) must recover x to machine precision."""
    rng = np.random.default_rng(0)
    x = rng.uniform(1e-5, 1e-2, size=20)
    recovered = unscale_rv(scale_rv(x))
    np.testing.assert_allclose(recovered, x, rtol=1e-12,
                               err_msg="Round-trip unscale_rv(scale_rv(x)) failed")


# ---------------------------------------------------------------------------
# scale_log_rv / unscale_log_rv
# ---------------------------------------------------------------------------

def test_scale_log_rv_lower_bound():
    """scale_log_rv(exp(-13)) should be approximately 0.0."""
    rv_low = np.array([np.exp(-13.0)])
    result = scale_log_rv(rv_low)
    assert abs(result[0]) < 1e-10, f"Expected ~0.0, got {result[0]}"


def test_scale_log_rv_upper_bound():
    """scale_log_rv(exp(-2.5)) should be approximately 1.0."""
    rv_high = np.array([np.exp(-2.5)])
    result = scale_log_rv(rv_high)
    assert abs(result[0] - 1.0) < 1e-10, f"Expected ~1.0, got {result[0]}"


def test_unscale_log_rv_roundtrip():
    """unscale_log_rv(scale_log_rv(x)) must recover x to machine precision."""
    rng = np.random.default_rng(1)
    x = np.exp(rng.uniform(-12.0, -3.0, size=20))  # realistic RV range
    recovered = unscale_log_rv(scale_log_rv(x))
    np.testing.assert_allclose(recovered, x, rtol=1e-12,
                               err_msg="Round-trip unscale_log_rv(scale_log_rv(x)) failed")


# ---------------------------------------------------------------------------
# clip_forecasts / compute_clip_threshold
# ---------------------------------------------------------------------------

def test_clip_forecasts_clips_correctly():
    """Values below the threshold should be raised; values above unchanged."""
    forecasts = np.array([0.001, 0.005, 0.05, 0.1])
    threshold = 0.01
    clipped = clip_forecasts(forecasts, threshold)
    assert clipped[0] == threshold, f"Expected {threshold}, got {clipped[0]}"
    assert clipped[1] == threshold, f"Expected {threshold}, got {clipped[1]}"
    assert clipped[2] == 0.05, f"Expected 0.05, got {clipped[2]}"
    assert clipped[3] == 0.1, f"Expected 0.1, got {clipped[3]}"


def test_clip_forecasts_no_clipping_needed():
    """When all forecasts exceed the threshold, values are unchanged."""
    forecasts = np.array([0.1, 0.2, 0.3])
    threshold = 0.05
    clipped = clip_forecasts(forecasts, threshold)
    np.testing.assert_array_equal(clipped, forecasts)


def test_compute_clip_threshold():
    """compute_clip_threshold returns 0.5 * min(rv_train[rv_train > 0])."""
    rv_train = np.array([0.0, 0.002, 0.004, 0.001, 0.0])
    expected = 0.5 * 0.001
    result = compute_clip_threshold(rv_train)
    assert abs(result - expected) < 1e-15, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# build_har_features
# ---------------------------------------------------------------------------

def test_build_har_features_shape():
    """Output shapes should be (n - burn_in, len(horizons)) and (n - burn_in,)."""
    n = 50
    horizons = (1, 5, 20)
    burn_in = 20
    z = np.random.default_rng(2).standard_normal(n)
    X, y = build_har_features(z, horizons, burn_in)
    assert X.shape == (n - burn_in, len(horizons)), (
        f"X.shape expected ({n - burn_in}, {len(horizons)}), got {X.shape}"
    )
    assert y.shape == (n - burn_in,), (
        f"y.shape expected ({n - burn_in},), got {y.shape}"
    )


def test_build_har_features_target():
    """y should equal z[burn_in:]."""
    n = 40
    horizons = (1, 5, 20)
    burn_in = 20
    z = np.arange(n, dtype=float)
    _, y = build_har_features(z, horizons, burn_in)
    np.testing.assert_array_equal(y, z[burn_in:])


def test_build_har_features_5day_average():
    """The 5-day feature column should be the backward-looking 5-day mean."""
    n = 40
    horizons = (1, 5, 20)
    burn_in = 20
    z = np.arange(n, dtype=float)
    X, _ = build_har_features(z, horizons, burn_in)
    # Feature index 1 = 5-day average
    h5_col = X[:, 1]
    for i, t in enumerate(range(burn_in, n)):
        expected = np.mean(z[t - 5 : t])
        assert abs(h5_col[i] - expected) < 1e-12, (
            f"At t={t}: expected {expected:.4f}, got {h5_col[i]:.4f}"
        )


def test_build_har_features_1day():
    """The 1-day feature should equal z[t-1] for each valid t."""
    n = 30
    horizons = (1, 5, 20)
    burn_in = 20
    z = np.arange(n, dtype=float)
    X, _ = build_har_features(z, horizons, burn_in)
    h1_col = X[:, 0]
    for i, t in enumerate(range(burn_in, n)):
        expected = z[t - 1]
        assert abs(h1_col[i] - expected) < 1e-12, (
            f"At t={t}: expected {expected:.4f}, got {h1_col[i]:.4f}"
        )


# ---------------------------------------------------------------------------
# proietti_lutkepohl_bias_correction
# ---------------------------------------------------------------------------

def test_proietti_correction_positive():
    """Bias-corrected values should always be strictly positive."""
    rng = np.random.default_rng(3)
    log_forecasts = rng.uniform(-10, -3, size=30)
    residuals = rng.normal(0, 0.3, size=100)
    corrected = proietti_lutkepohl_bias_correction(log_forecasts, residuals)
    assert np.all(corrected > 0), "Bias-corrected forecasts should be positive"


def test_proietti_correction_exceeds_naive():
    """
    With non-trivial residual variance the bias-corrected value should exceed
    exp(mu) (the naive inverse of the log-normal), since exp(0.5 * σ²) > 1.
    """
    mu = np.array([-7.0, -6.0, -5.0])
    residuals = np.random.default_rng(4).normal(0, 0.5, size=200)
    corrected = proietti_lutkepohl_bias_correction(mu, residuals)
    naive = np.exp(mu)
    assert np.all(corrected > naive), (
        "Bias-corrected values should exceed naive exp(mu) when σ² > 0"
    )
