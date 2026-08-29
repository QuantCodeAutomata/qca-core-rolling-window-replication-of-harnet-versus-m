"""Unit tests for src/har_model.py (HARBaseline)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.har_model import HARBaseline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_RV_TRAIN = np.exp(_RNG.normal(-7.0, 0.3, size=200))  # realistic RV, n=200


def _fitted_model(fit_type: str, horizons: tuple = (1, 5, 20)) -> HARBaseline:
    m = HARBaseline(fit_type=fit_type, horizons=horizons)
    m.fit(_RV_TRAIN)
    return m


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def test_har_baseline_ols_fit():
    """OLS fit should succeed and produce len(horizons)+1 coefficients."""
    m = _fitted_model("OLS")
    assert m.coef_ is not None, "coef_ should not be None after fit"
    assert len(m.coef_) == len((1, 5, 20)) + 1, (
        f"Expected {len((1,5,20))+1} params, got {len(m.coef_)}"
    )


def test_har_baseline_wls_fit():
    """WLS fit should succeed and produce len(horizons)+1 coefficients."""
    m = _fitted_model("WLS")
    assert m.coef_ is not None, "coef_ should not be None after WLS fit"
    assert len(m.coef_) == len((1, 5, 20)) + 1


def test_har_baseline_logols_fit():
    """log-OLS fit should succeed and produce len(horizons)+1 coefficients."""
    m = _fitted_model("log-OLS")
    assert m.coef_ is not None, "coef_ should not be None after log-OLS fit"
    assert len(m.coef_) == len((1, 5, 20)) + 1


def test_har_baseline_coef_finite():
    """Fitted coefficients must contain no NaN or Inf for all three fit types."""
    for fit_type in ("OLS", "WLS", "log-OLS"):
        m = _fitted_model(fit_type)
        assert np.all(np.isfinite(m.coef_)), (
            f"coef_ contains non-finite values for fit_type='{fit_type}'"
        )


def test_har_baseline_coef_length_har80():
    """For HAR80 (5 horizons) coefficients should have length 6."""
    m = HARBaseline("OLS", horizons=(1, 5, 20, 40, 80))
    m.fit(_RV_TRAIN)
    assert len(m.coef_) == 6, f"Expected 6 params for HAR80, got {len(m.coef_)}"


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def test_har_predictions_shape():
    """predict() should return an array of length n - burn_in."""
    horizons = (1, 5, 20)
    burn_in = max(horizons)
    m = _fitted_model("OLS", horizons=horizons)
    preds = m.predict(_RV_TRAIN)
    expected_len = len(_RV_TRAIN) - burn_in
    assert len(preds) == expected_len, (
        f"Expected length {expected_len}, got {len(preds)}"
    )


def test_har_predictions_shape_with_start_idx():
    """predict(rv, start_idx=n_train) should slice correctly."""
    horizons = (1, 5, 20)
    burn_in = max(horizons)
    m = _fitted_model("OLS", horizons=horizons)
    n_train = 150
    rv_full = np.concatenate([_RV_TRAIN, np.exp(_RNG.normal(-7, 0.3, 30))])
    preds = m.predict(rv_full, start_idx=n_train)
    expected_len = len(rv_full) - n_train
    assert len(preds) == expected_len, (
        f"Expected length {expected_len}, got {len(preds)}"
    )


def test_har_rv_scale_predictions_positive():
    """predict_rv_scale() should return strictly positive values."""
    for fit_type in ("OLS", "WLS", "log-OLS"):
        m = _fitted_model(fit_type)
        preds_rv = m.predict_rv_scale(_RV_TRAIN)
        assert np.all(preds_rv > 0), (
            f"predict_rv_scale returned non-positive values for fit_type='{fit_type}'"
        )


def test_har_rv_scale_predictions_finite():
    """predict_rv_scale() should contain no NaN or Inf."""
    for fit_type in ("OLS", "WLS", "log-OLS"):
        m = _fitted_model(fit_type)
        preds_rv = m.predict_rv_scale(_RV_TRAIN)
        assert np.all(np.isfinite(preds_rv)), (
            f"predict_rv_scale returned non-finite values for fit_type='{fit_type}'"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_invalid_fit_type():
    """Constructing HARBaseline with an unknown fit_type should raise ValueError."""
    with pytest.raises(ValueError, match="fit_type must be one of"):
        HARBaseline(fit_type="INVALID")


def test_predict_before_fit():
    """Calling predict() before fit() should raise RuntimeError."""
    m = HARBaseline("OLS")
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict(_RV_TRAIN)


def test_predict_rv_scale_before_fit():
    """Calling predict_rv_scale() before fit() should raise RuntimeError."""
    m = HARBaseline("WLS")
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict_rv_scale(_RV_TRAIN)
