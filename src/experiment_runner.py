"""
Core pipeline used by all experiments.

Provides:
  - run_window_config : fit HAR baseline + HARNet for one (train, test) split
  - glorot_init_harnet : Glorot uniform re-initialisation (used by Exp2)
  - compute_all_test_metrics : MAE / MSE / QLIKE from two rv arrays

Scaling contract
----------------
HARNet always operates on the *scaled z domain*.

  OLS / WLS  : z = scale_rv(rv)      ; rv = unscale_rv(z)
  log-OLS    : z = scale_log_rv(rv)  ; back-transform uses
               Proietti-Lütkepohl bias correction after converting
               z → log(rv) via  log_rv = z * LOG_RANGE − LOG_SHIFT
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.har_model import HARBaseline
from src.harnet import HARNet, HARNetTrainer
from src.losses import LOSS_FN_NP
from src.transforms import (
    LOG_RANGE,
    LOG_SHIFT,
    clip_forecasts,
    compute_clip_threshold,
    proietti_lutkepohl_bias_correction,
    scale_log_rv,
    scale_rv,
    unscale_rv,
)
from src.utils import set_seeds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def glorot_init_harnet(net: HARNet) -> None:
    """Apply Glorot uniform init to all HARNet parameters (for Exp2 stability test)."""
    with torch.no_grad():
        for param in net.parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                # xavier_uniform_ requires ≥ 2-D; expand, init, collapse
                tmp = param.data.unsqueeze(0)
                nn.init.xavier_uniform_(tmp)
                param.data.copy_(tmp.squeeze(0))


def compute_all_test_metrics(
    rv_true: np.ndarray,
    rv_pred: np.ndarray,
) -> dict:
    """
    Compute MAE, MSE, and QLIKE between two rv arrays.

    Returns
    -------
    dict with keys 'MAE', 'MSE', 'QLIKE' (float values).
    """
    return {name: fn(rv_true, rv_pred) for name, fn in LOSS_FN_NP.items()}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_window_config(
    rv_train: np.ndarray,
    rv_test: np.ndarray,
    horizons: tuple,
    fit_type: str,  # 'OLS', 'WLS', 'log-OLS'
    loss_name: str,  # 'MAE', 'MSE', 'QLIKE'
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Run one complete train/evaluate cycle for a HAR baseline and HARNet.

    Steps
    -----
    1. Scale rv_train and rv_full to z domain (scale_rv or scale_log_rv).
    2. Fit HARBaseline on rv_train.
    3. Init HARNet from baseline coefficients; assert pre-train parity < 1e-5.
    4. Train HARNet on z_train for n_iter iterations.
    5. Predict on z_full starting from len(rv_train) → n_test z-scale forecasts.
    6. Unscale to rv domain (+ Proietti-Lütkepohl for log-OLS).
    7. Clip forecasts; compute MAE / MSE / QLIKE for both models.

    Parameters
    ----------
    rv_train, rv_test : np.ndarray
        Raw realized variance arrays (not transformed).
    horizons : tuple of int
        HAR lag horizons, e.g. (1, 5, 20) or (1, 5, 20, 40, 80).
    fit_type : str
        'OLS', 'WLS', or 'log-OLS' — governs transform and baseline fit.
    loss_name : str
        'MAE', 'MSE', or 'QLIKE' — loss used during HARNet training.
    n_iter : int
        Number of gradient update steps for HARNet.
    seed : int
        Master seed for all RNGs.

    Returns
    -------
    dict
        Keys: baseline_coef, pre_train_max_diff,
              baseline_mae/mse/qlike, harnet_mae/mse/qlike,
              relative_mae/mse/qlike,
              baseline_forecasts_rv, harnet_forecasts_rv,
              rv_test, train_losses.
    """
    set_seeds(seed)

    burn_in = max(horizons)
    rv_full = np.concatenate([rv_train, rv_test])

    # ------------------------------------------------------------------
    # 1. Scale to z domain
    # ------------------------------------------------------------------
    if fit_type == "log-OLS":
        z_train = scale_log_rv(rv_train)
        z_full = scale_log_rv(rv_full)
    else:
        z_train = scale_rv(rv_train)
        z_full = scale_rv(rv_full)

    # ------------------------------------------------------------------
    # 2. Fit HAR baseline on rv_train
    # ------------------------------------------------------------------
    baseline = HARBaseline(fit_type=fit_type, horizons=horizons)
    baseline.fit(rv_train)

    # ------------------------------------------------------------------
    # 3. Init HARNet from baseline; verify pre-train parity
    # ------------------------------------------------------------------
    net = HARNet(horizons=horizons)
    net.init_from_baseline(baseline.coef_)

    # baseline.predict returns z-scale predictions for all valid positions
    baseline_z_pretrain = baseline.predict(rv_train)  # shape: (len(rv_train) - burn_in,)

    trainer = HARNetTrainer(
        model=net,
        loss_name=loss_name,
        n_iter=n_iter,
        seed=seed,
    )
    # trainer.predict does not modify parameters — safe before training
    harnet_z_pretrain = trainer.predict(z_train, start_idx=burn_in)

    pre_train_max_diff = float(np.max(np.abs(baseline_z_pretrain - harnet_z_pretrain)))
    assert pre_train_max_diff < 1e-5, (
        f"Pre-train max abs diff {pre_train_max_diff:.2e} ≥ 1e-5 — "
        "HARNet is not equivalent to baseline at initialisation."
    )

    # ------------------------------------------------------------------
    # 4. Train HARNet on z_train
    # ------------------------------------------------------------------
    train_losses = trainer.train(z_train)

    # ------------------------------------------------------------------
    # 5. Forecast for test period using full series context
    # ------------------------------------------------------------------
    start_idx = len(rv_train)

    # baseline.predict internally transforms rv_full → z, returning z-scale preds
    baseline_z_test = baseline.predict(rv_full, start_idx=start_idx)  # (n_test,)
    harnet_z_test = trainer.predict(z_full, start_idx=start_idx)       # (n_test,)

    # ------------------------------------------------------------------
    # 6. Convert z-scale predictions → rv scale
    # ------------------------------------------------------------------
    if fit_type in ("OLS", "WLS"):
        baseline_forecasts_rv = unscale_rv(baseline_z_test)
        harnet_forecasts_rv = unscale_rv(harnet_z_test)
    else:  # log-OLS
        # z → log(rv): inverse of (log(rv) + LOG_SHIFT) / LOG_RANGE
        log_rv_baseline = baseline_z_test * LOG_RANGE - LOG_SHIFT
        log_rv_harnet = harnet_z_test * LOG_RANGE - LOG_SHIFT
        # Residuals stored in scaled-log space; convert to log(rv) space for σ² estimate
        residuals_log = baseline.residuals_ * LOG_RANGE
        baseline_forecasts_rv = proietti_lutkepohl_bias_correction(log_rv_baseline, residuals_log)
        harnet_forecasts_rv = proietti_lutkepohl_bias_correction(log_rv_harnet, residuals_log)

    # ------------------------------------------------------------------
    # 7. Clip forecasts to avoid near-zero predictions
    # ------------------------------------------------------------------
    clip_threshold = compute_clip_threshold(rv_train)
    baseline_forecasts_rv = clip_forecasts(baseline_forecasts_rv, clip_threshold)
    harnet_forecasts_rv = clip_forecasts(harnet_forecasts_rv, clip_threshold)

    # ------------------------------------------------------------------
    # 8. Compute all metrics on rv scale
    # ------------------------------------------------------------------
    b_mae = LOSS_FN_NP["MAE"](rv_test, baseline_forecasts_rv)
    b_mse = LOSS_FN_NP["MSE"](rv_test, baseline_forecasts_rv)
    b_qlike = LOSS_FN_NP["QLIKE"](rv_test, baseline_forecasts_rv)
    h_mae = LOSS_FN_NP["MAE"](rv_test, harnet_forecasts_rv)
    h_mse = LOSS_FN_NP["MSE"](rv_test, harnet_forecasts_rv)
    h_qlike = LOSS_FN_NP["QLIKE"](rv_test, harnet_forecasts_rv)

    return {
        "baseline_coef": baseline.coef_,
        "pre_train_max_diff": pre_train_max_diff,
        "baseline_mae": b_mae,
        "baseline_mse": b_mse,
        "baseline_qlike": b_qlike,
        "harnet_mae": h_mae,
        "harnet_mse": h_mse,
        "harnet_qlike": h_qlike,
        "relative_mae": h_mae / b_mae,
        "relative_mse": h_mse / b_mse,
        "relative_qlike": h_qlike / b_qlike,
        "baseline_forecasts_rv": baseline_forecasts_rv,
        "harnet_forecasts_rv": harnet_forecasts_rv,
        "rv_test": rv_test,
        "train_losses": train_losses,
    }
