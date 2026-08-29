"""
HAR baseline models: OLS, WLS, and log-OLS.

All three variants share the same HAR-RV regressor structure (intercept plus
backward-looking averages over horizons 1, 5, and 20 by default) but differ
in the transformation applied to RV before fitting and how predictions are
converted back to the RV level.

OLS
    Fit on linearly scaled RV (×1 000).  Forecasts are unscaled with
    ``unscale_rv``.

WLS
    Same as OLS but with linearly increasing observation weights so that
    more recent observations receive higher influence:
    ``w_i = (i + 1) / sum(1 … n)``.

log-OLS
    Fit on log-and-linearly-scaled RV (``scale_log_rv``).  Forecasts are
    converted back to the RV level with the Proietti-Lütkepohl (2013)
    bias correction:  E[RV] = exp(log_forecast + 0.5 × σ²_residuals).
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from src.transforms import (
    LOG_RANGE,
    build_har_features,
    proietti_lutkepohl_bias_correction,
    scale_log_rv,
    scale_rv,
    unscale_rv,
)

_VALID_FIT_TYPES = ("OLS", "WLS", "log-OLS")


class HARBaseline:
    """
    HAR-RV baseline model supporting OLS, WLS, and log-OLS fitting.

    Parameters
    ----------
    fit_type : str
        One of ``'OLS'``, ``'WLS'``, or ``'log-OLS'``.
    horizons : tuple of int
        Lag horizons used to build the HAR regressors.  Default ``(1, 5, 20)``.

    Attributes
    ----------
    coef_ : np.ndarray, shape (1 + len(horizons),)
        Fitted coefficients: intercept first, then one per horizon.
    residuals_ : np.ndarray
        In-sample residuals on the transformed scale used during fitting.
    """

    def __init__(
        self,
        fit_type: str,
        horizons: tuple = (1, 5, 20),
    ) -> None:
        if fit_type not in _VALID_FIT_TYPES:
            raise ValueError(
                f"fit_type must be one of {_VALID_FIT_TYPES}, got '{fit_type}'"
            )
        self.fit_type = fit_type
        self.horizons = horizons
        self.coef_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, rv_train: np.ndarray) -> None:
        """
        Fit the HAR model on *rv_train*.

        Parameters
        ----------
        rv_train : np.ndarray
            Daily realized variance series (raw, not transformed).

        Raises
        ------
        AssertionError
            If the fitted coefficients contain non-finite values.
        """
        # 1. Transform
        if self.fit_type == "log-OLS":
            z = scale_log_rv(rv_train)
        else:
            z = scale_rv(rv_train)

        # 2. Build HAR features
        burn_in = max(self.horizons)
        X, y = build_har_features(z, self.horizons, burn_in)
        X_const = sm.add_constant(X, has_constant="add")

        # 3. Fit
        n_obs = len(y)
        if self.fit_type == "WLS":
            # Linearly increasing weights: w_i = (i+1) / sum(1…n)
            raw_weights = np.arange(1, n_obs + 1, dtype=float)
            weights = raw_weights / raw_weights.sum()
            result = sm.WLS(y, X_const, weights=weights).fit()
        else:
            result = sm.OLS(y, X_const).fit()

        self.coef_ = result.params          # intercept, beta1, …, betaH
        self.residuals_ = result.resid      # on the transformed (scaled) space

        assert np.all(np.isfinite(self.coef_)), (
            "Fitted coefficients contain non-finite values."
        )

    # ------------------------------------------------------------------
    # Prediction (transformed scale)
    # ------------------------------------------------------------------

    def predict(
        self,
        rv_series: np.ndarray,
        start_idx: int | None = None,
    ) -> np.ndarray:
        """
        Generate one-step-ahead predictions on the **transformed** scale.

        Parameters
        ----------
        rv_series : np.ndarray
            Full (or partial) RV series to generate predictions from.
        start_idx : int or None
            If given, return only predictions for positions ``start_idx`` …
            ``len(rv_series) - 1``.  Must be ≥ ``max(self.horizons)``.

        Returns
        -------
        np.ndarray
            Predictions on the scaled (fitted) space.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.  Call .fit() first.")

        if self.fit_type == "log-OLS":
            z = scale_log_rv(rv_series)
        else:
            z = scale_rv(rv_series)

        burn_in = max(self.horizons)
        X, _ = build_har_features(z, self.horizons, burn_in)
        X_const = sm.add_constant(X, has_constant="add")

        y_hat = X_const @ self.coef_

        if start_idx is not None:
            offset = start_idx - burn_in
            if offset < 0:
                raise ValueError(
                    f"start_idx ({start_idx}) must be >= burn_in ({burn_in})."
                )
            y_hat = y_hat[offset:]

        return y_hat

    # ------------------------------------------------------------------
    # Prediction (RV scale)
    # ------------------------------------------------------------------

    def predict_rv_scale(
        self,
        rv_series: np.ndarray,
        rv_train: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate one-step-ahead predictions on the **raw RV** scale.

        For ``OLS`` and ``WLS`` the scaled predictions are simply divided by
        ``RV_SCALE`` (1 000).  For ``log-OLS`` the Proietti-Lütkepohl (2013)
        bias correction is applied:

            E[RV] = exp(log_forecast) × exp(0.5 × σ²_log)

        where σ²_log is estimated from the in-sample log-space residuals.

        Parameters
        ----------
        rv_series : np.ndarray
            Full RV series used to construct features.
        rv_train : np.ndarray or None
            Unused; kept for API symmetry with HARNet predictors.

        Returns
        -------
        np.ndarray
            Forecasts in raw RV units.
        """
        y_hat_scaled = self.predict(rv_series)

        if self.fit_type in ("OLS", "WLS"):
            return unscale_rv(y_hat_scaled)

        # log-OLS: convert scaled-log predictions → log(RV) → bias-corrected RV
        log_y_hat = y_hat_scaled * LOG_RANGE - 13.0  # unscale_log_rv without exp
        # Residuals are in the scaled-log space; convert to log(RV) space for
        # the bias correction (multiply by LOG_RANGE).
        residuals_log = self.residuals_ * LOG_RANGE
        return proietti_lutkepohl_bias_correction(log_y_hat, residuals_log)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self.coef_ is not None
        return (
            f"HARBaseline(fit_type='{self.fit_type}', "
            f"horizons={self.horizons}, fitted={fitted})"
        )
