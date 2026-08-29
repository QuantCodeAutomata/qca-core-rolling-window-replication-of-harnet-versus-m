"""
Data transformation utilities for HAR and HARNet models.

Two scaling conventions are supported:

``scale_rv`` / ``unscale_rv``
    Linear scaling of the raw RV level: multiply by 1 000 so that typical
    values (RV ~ 1e-3) map to O(1).

``scale_log_rv`` / ``unscale_log_rv``
    Log-transform followed by a linear shift/scale that maps the realistic
    range log(RV) ∈ [-13, -2.5] to approximately [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union

RV_SCALE = 1000.0   # 0 -> 0, 1e-3 -> 1
LOG_SHIFT = 13.0    # -13 maps to 0
LOG_RANGE = 10.5    # span: -13 to -2.5


# ---------------------------------------------------------------------------
# Linear RV scaling
# ---------------------------------------------------------------------------

def scale_rv(rv: np.ndarray) -> np.ndarray:
    """Linear scaling of raw RV: 0->0, 1e-3->1.  x_scaled = 1000 * rv."""
    return rv * RV_SCALE


def unscale_rv(rv_scaled: np.ndarray) -> np.ndarray:
    """Inverse of :func:`scale_rv`."""
    return rv_scaled / RV_SCALE


# ---------------------------------------------------------------------------
# Log-space scaling
# ---------------------------------------------------------------------------

def scale_log_rv(rv: np.ndarray) -> np.ndarray:
    """
    Log-transform then linear scale so that log(rv) ∈ [-13, -2.5] → [0, 1].

        x_scaled = (log(rv) + 13) / 10.5

    Parameters
    ----------
    rv : np.ndarray
        Raw realized variance values (strictly positive).

    Returns
    -------
    np.ndarray
        Scaled values, typically in [0, 1] for realistic RV.
    """
    log_rv = np.log(rv)
    return (log_rv + LOG_SHIFT) / LOG_RANGE


def unscale_log_rv(rv_scaled: np.ndarray) -> np.ndarray:
    """
    Inverse of :func:`scale_log_rv`: recover RV from the scaled log space.

        rv = exp(rv_scaled * 10.5 - 13)

    Parameters
    ----------
    rv_scaled : np.ndarray
        Scaled values produced by :func:`scale_log_rv`.

    Returns
    -------
    np.ndarray
        Raw realized variance values.
    """
    log_rv = rv_scaled * LOG_RANGE - LOG_SHIFT
    return np.exp(log_rv)


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------

def proietti_lutkepohl_bias_correction(
    sigma2_hat_log: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """
    Apply Proietti-Lütkepohl (2013) bias correction to convert log-scale
    forecasts to the RV level.

    For a log-normal model ``log(RV) ~ N(mu, sigma2)``:

        E[RV] = exp(mu + 0.5 * sigma2)

    so:

        bias_corrected = exp(log_forecast) * exp(0.5 * sigma2_residuals)

    Parameters
    ----------
    sigma2_hat_log : np.ndarray
        Predicted **log** RV values (not the scaled version; units of log-RV).
    residuals : np.ndarray
        OLS residuals from the log-scale fit, used to estimate sigma2.

    Returns
    -------
    np.ndarray
        Bias-corrected RV-level forecasts.
    """
    sigma2 = np.var(residuals, ddof=1)
    return np.exp(sigma2_hat_log) * np.exp(0.5 * sigma2)


# ---------------------------------------------------------------------------
# Forecast clipping
# ---------------------------------------------------------------------------

def clip_forecasts(forecasts: np.ndarray, clip_threshold: float) -> np.ndarray:
    """Clip forecasts to be at least *clip_threshold* (avoids near-zero predictions)."""
    return np.maximum(forecasts, clip_threshold)


def compute_clip_threshold(rv_train: np.ndarray) -> float:
    """Compute clipping threshold as 0.5 * min(rv_train > 0)."""
    return 0.5 * np.min(rv_train[rv_train > 0])


# ---------------------------------------------------------------------------
# HAR feature construction
# ---------------------------------------------------------------------------

def build_har_features(
    z: np.ndarray,
    horizons: tuple,
    burn_in: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build HAR regressors from a transformed daily series *z*.

    For each horizon *h* in *horizons* and each valid time step *t*, the
    corresponding feature is the backward-looking mean over the *h* most
    recent values::

        X[t, i] = mean(z[t-h : t])   for horizon horizons[i]

    The first *burn_in* observations are consumed as context and excluded from
    the returned arrays.

    Parameters
    ----------
    z : np.ndarray
        Transformed (scaled) daily RV series of length *n*.
    horizons : tuple of int
        Lag horizons, e.g. ``(1, 5, 20)`` or ``(1, 5, 20, 40, 80)``.
    burn_in : int
        Number of leading observations to skip; should equal ``max(horizons)``.

    Returns
    -------
    X : np.ndarray, shape (n - burn_in, len(horizons))
        Feature matrix.
    y : np.ndarray, shape (n - burn_in,)
        Target series ``z[burn_in:]``.
    """
    max_h = max(horizons)
    n = len(z)
    valid_start = max_h
    n_valid = n - valid_start

    X = np.zeros((n_valid, len(horizons)))
    for i, h in enumerate(horizons):
        for t in range(valid_start, n):
            X[t - valid_start, i] = np.mean(z[t - h : t])

    y = z[valid_start:]
    return X, y
