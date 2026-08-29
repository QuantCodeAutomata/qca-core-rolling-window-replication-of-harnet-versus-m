"""
Synthetic realized variance data generation mimicking the Oxford-Man Institute
realized library for SPX, FTSE, and DJI.

The data-generating process is a HAR model in log-space:
    log(RV_t) = c + b1*log(RV_{t-1})
              + b5*mean(log(RV_{t-5:t}))
              + b20*mean(log(RV_{t-20:t}))
              + eps,  eps ~ N(0, sigma^2)

with b1=0.3, b5=0.3, b20=0.2, sigma=0.3.  The constant c is chosen so that
the stationary mean of log(RV) is approximately -7, giving RV ~ exp(-7) ~ 9e-4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# HAR process coefficients
_BETA_DAILY = 0.30
_BETA_WEEKLY = 0.30
_BETA_MONTHLY = 0.20
_SIGMA_EPS = 0.30

# Stationary mean target: mu = c / (1 - b1 - b5 - b20)
# => c = mu * (1 - 0.8) = -7 * 0.2 = -1.4
_CONST = -1.4
_MU_TARGET = _CONST / (1.0 - _BETA_DAILY - _BETA_WEEKLY - _BETA_MONTHLY)  # ≈ -7.0

_TICKER_SEEDS = {"SPX": 42, "FTSE": 43, "DJI": 44}
_WARMUP = 500


def generate_rv_series(
    n_days: int,
    seed: int,
    base_rv: float = np.exp(-7.0),
    persistence: float = 0.8,
) -> np.ndarray:
    """
    Generate a simple log-normal AR(1) realized variance series.

    The model is:
        log(RV_t) = c_ar1 + persistence * log(RV_{t-1}) + eps,
        eps ~ N(0, sigma^2),  sigma = 0.3.

    The constant c_ar1 is chosen so the stationary mean of log(RV) equals
    log(base_rv).

    Parameters
    ----------
    n_days : int
        Number of trading days to return.
    seed : int
        NumPy random seed for reproducibility.
    base_rv : float
        Target long-run mean level of RV (not log-RV).
    persistence : float
        AR(1) coefficient for log(RV).

    Returns
    -------
    np.ndarray
        Array of RV values with shape (n_days,).
    """
    rng = np.random.default_rng(seed)
    mu_log = np.log(base_rv)
    c_ar1 = mu_log * (1.0 - persistence)

    total = n_days + _WARMUP
    eps = rng.normal(0.0, _SIGMA_EPS, size=total)

    log_rv = np.empty(total)
    log_rv[0] = mu_log
    for t in range(1, total):
        log_rv[t] = c_ar1 + persistence * log_rv[t - 1] + eps[t]

    return np.exp(log_rv[_WARMUP:])


def _generate_har_log_rv(n_target: int, seed: int) -> np.ndarray:
    """
    Internal helper: generate log(RV) from a HAR(1,5,20) process.

    Returns log-RV array of length n_target (after warmup).
    """
    rng = np.random.default_rng(seed)
    total = n_target + _WARMUP
    eps = rng.normal(0.0, _SIGMA_EPS, size=total)

    log_rv = np.empty(total)
    # Initialise warmup region at the stationary mean
    log_rv[:20] = _MU_TARGET

    for t in range(20, total):
        daily = log_rv[t - 1]
        weekly = np.mean(log_rv[t - 5 : t])
        monthly = np.mean(log_rv[t - 20 : t])
        log_rv[t] = (
            _CONST
            + _BETA_DAILY * daily
            + _BETA_WEEKLY * weekly
            + _BETA_MONTHLY * monthly
            + eps[t]
        )

    return log_rv[_WARMUP:]


def load_oxford_man_data(
    tickers: List[str] = ("SPX", "FTSE", "DJI"),
    start: str = "2001-08-29",
    end: str = "2020-12-31",
    data_dir: Optional[str] = None,
) -> Dict[str, pd.Series]:
    """
    Generate synthetic daily realized variance data mimicking the Oxford-Man
    Institute realized library.

    The data-generating process is a HAR model in log-space (see module
    docstring).  No internet access is required; all data are generated
    synthetically.

    Parameters
    ----------
    tickers : list of str
        Asset identifiers.  Seeds are fixed per ticker:
        SPX=42, FTSE=43, DJI=44.  Unknown tickers derive their seed from
        the sum of their ASCII codes.
    start : str
        First business day (inclusive), default ``"2001-08-29"``.
    end : str
        Last business day (inclusive), default ``"2020-12-31"``.
    data_dir : str or None
        Unused (kept for API compatibility with a real data loader).

    Returns
    -------
    dict
        Mapping ``ticker -> pd.Series`` of daily RV values with a
        ``DatetimeIndex`` of business days.
    """
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)

    result: Dict[str, pd.Series] = {}
    for ticker in tickers:
        seed = _TICKER_SEEDS.get(ticker, sum(ord(c) for c in ticker))
        log_rv = _generate_har_log_rv(n_days, seed)
        rv = np.exp(log_rv)
        result[ticker] = pd.Series(rv, index=dates, name=ticker)

    return result


def _generate_semivariance(
    rv: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate negative realized semivariance (rv_minus) and signed jump variation.

    rv_minus is modelled as:
        rv_minus_t = clip(0.40 * RV_t + noise_t, 0.01*RV_t, 0.99*RV_t)
    where noise_t ~ N(0, (0.05*RV_t)^2).

    signed_jump = RS_plus - RS_minus
                = (RV - rv_minus) - rv_minus
                = RV - 2 * rv_minus

    Parameters
    ----------
    rv : np.ndarray
        Daily RV series.
    seed : int
        Random seed for noise generation.

    Returns
    -------
    rv_minus : np.ndarray
    signed_jump : np.ndarray
    """
    rng = np.random.default_rng(seed + 1000)
    noise = rng.normal(0.0, 0.05 * rv)
    rv_minus = np.clip(0.40 * rv + noise, 0.01 * rv, 0.99 * rv)
    signed_jump = rv - 2.0 * rv_minus
    return rv_minus, signed_jump


def load_extended_data(
    tickers: List[str] = ("SPX", "FTSE", "DJI"),
    start: str = "2001-08-29",
    end: str = "2020-12-31",
    data_dir: Optional[str] = None,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Generate synthetic RV, negative semivariance, and signed jump variation.

    Returns
    -------
    dict with keys ``'rv'``, ``'rv_minus'``, ``'signed_jump'``, each mapping
    ``ticker -> pd.Series``.
    """
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)

    rv_dict: Dict[str, pd.Series] = {}
    rv_minus_dict: Dict[str, pd.Series] = {}
    signed_jump_dict: Dict[str, pd.Series] = {}

    for ticker in tickers:
        seed = _TICKER_SEEDS.get(ticker, sum(ord(c) for c in ticker))
        log_rv = _generate_har_log_rv(n_days, seed)
        rv = np.exp(log_rv)

        rv_minus_arr, signed_jump_arr = _generate_semivariance(rv, seed)

        rv_dict[ticker] = pd.Series(rv, index=dates, name=ticker)
        rv_minus_dict[ticker] = pd.Series(rv_minus_arr, index=dates, name=ticker)
        signed_jump_dict[ticker] = pd.Series(signed_jump_arr, index=dates, name=ticker)

    return {
        "rv": rv_dict,
        "rv_minus": rv_minus_dict,
        "signed_jump": signed_jump_dict,
    }
