"""
Exp 5 — Extended-input HARNet (RV + RS- + ΔJ).

For each of the 3 indexes and 15 rolling windows, an extended HAR-OLS baseline
and a HARNetExtended are evaluated with two architectures: HAR20 and HAR80.

Extended HAR regressors for HAR20 (horizons = (1, 5, 20))
----------------------------------------------------------
  [RV_1, RV_5, RV_20, RS-_1, RS-_5, RS-_20, ΔJ_1]
  where _h denotes the h-day backward average (scale_rv applied to RV and RS-;
  ΔJ divided by RV_SCALE to normalise to the same scale).

Pre-training equivalence is verified: the HARNetExtended predictions (with
uniform filters and OLS output weights) must match the extended OLS baseline
predictions to within 1e-4 absolute error.

Outputs
-------
  results/exp5_extended_results.csv
  results/exp5_relative_errors.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from data.generate_extended_data import load_extended_data
from src.harnet_extended import HARNetExtended, HARNetExtendedTrainer
from src.losses import LOSS_FN_NP
from src.rolling_windows import WINDOWS, get_window_dates, split_series
from src.transforms import (
    RV_SCALE,
    build_har_features,
    clip_forecasts,
    compute_clip_threshold,
    scale_rv,
    unscale_rv,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEXES = ["SPX", "FTSE", "DJI"]
ARCH_LIST: list[tuple] = [(1, 5, 20), (1, 5, 20, 40, 80)]
METRICS = ["MAE", "MSE", "QLIKE"]
N_ITER = 1000
LR = 1e-4
BATCH_SIZE = 4
SEG_LEN = 5
SEED = 42
EQUIV_TOL = 1e-4

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_extended_har_features(
    rv: np.ndarray,
    rs_neg: np.ndarray,
    delta_j: np.ndarray,
    horizons: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build extended HAR regressors consistent with HARNetExtended.forward.

    Parameters
    ----------
    rv, rs_neg : np.ndarray
        Raw RV and negative semivariance series.
    delta_j : np.ndarray
        Signed jump variation series (can be negative).
    horizons : tuple of int

    Returns
    -------
    X : np.ndarray, shape (n_valid, 2 * len(horizons) + 1)
        Feature matrix: [rv_h features, rs_h features, ΔJ lag-1 / RV_SCALE].
    y : np.ndarray, shape (n_valid,)
        Target = scale_rv(rv)[burn_in:].
    """
    burn_in = max(horizons)
    z_rv = scale_rv(rv)
    z_rs = scale_rv(rs_neg)

    X_rv, y = build_har_features(z_rv, horizons, burn_in)
    X_rs, _ = build_har_features(z_rs, horizons, burn_in)

    n = len(rv)
    n_valid = n - burn_in
    # ΔJ at lag 1: delta_j[t-1] for t = burn_in … n-1
    dj_feat = delta_j[burn_in - 1 : n - 1] / RV_SCALE  # (n_valid,)

    X = np.column_stack([X_rv, X_rs, dj_feat])  # (n_valid, 2*n_horizons+1)
    return X, y


def fit_extended_ols(
    rv_train: np.ndarray,
    rs_train: np.ndarray,
    dj_train: np.ndarray,
    horizons: tuple,
) -> np.ndarray:
    """
    Fit extended HAR-OLS and return coefficient vector.

    Returns
    -------
    coef : np.ndarray, shape (1 + 2 * len(horizons) + 1,)
        [intercept, rv_betas..., rs_betas..., jump_beta]
    """
    X, y = build_extended_har_features(rv_train, rs_train, dj_train, horizons)
    X_const = sm.add_constant(X, has_constant="add")
    result = sm.OLS(y, X_const).fit()
    return result.params  # (1 + 2*n_horizons + 1,)


def extended_ols_test_preds(
    rv_train: np.ndarray,
    rs_train: np.ndarray,
    dj_train: np.ndarray,
    rv_test: np.ndarray,
    rs_test: np.ndarray,
    dj_test: np.ndarray,
    horizons: tuple,
    coef: np.ndarray,
) -> np.ndarray:
    """
    Predict test-period values with the extended OLS baseline and unscale to RV.

    Returns
    -------
    np.ndarray
        Clipped RV-scale predictions for the test period.
    """
    rv_full = np.concatenate([rv_train, rv_test])
    rs_full = np.concatenate([rs_train, rs_test])
    dj_full = np.concatenate([dj_train, dj_test])

    X_full, _ = build_extended_har_features(rv_full, rs_full, dj_full, horizons)
    X_const = sm.add_constant(X_full, has_constant="add")
    preds_scaled = X_const @ coef  # all valid positions (scaled z)

    burn_in = max(horizons)
    n_train = len(rv_train)
    # Predictions from index n_train onwards
    test_preds_scaled = preds_scaled[n_train - burn_in :]
    test_preds_rv = unscale_rv(test_preds_scaled)

    clip_th = compute_clip_threshold(rv_train)
    return clip_forecasts(test_preds_rv, clip_th)


# ---------------------------------------------------------------------------
# Equivalence verification
# ---------------------------------------------------------------------------

def verify_equivalence(
    net: HARNetExtended,
    rv_train: np.ndarray,
    rs_train: np.ndarray,
    dj_train: np.ndarray,
    coef: np.ndarray,
    horizons: tuple,
    tol: float = EQUIV_TOL,
) -> float:
    """
    Compare HARNetExtended (post-init) vs extended OLS on the training series.

    Returns the maximum absolute difference.  Raises AssertionError if > tol.
    """
    X, _ = build_extended_har_features(rv_train, rs_train, dj_train, horizons)
    X_const = sm.add_constant(X, has_constant="add")
    baseline_preds = X_const @ coef  # (n_valid,) in scaled z

    z_rv = scale_rv(rv_train)
    z_rs = scale_rv(rs_train)
    dj_norm = dj_train / RV_SCALE

    rv_t = torch.tensor(z_rv, dtype=torch.float32).unsqueeze(0)
    rs_t = torch.tensor(z_rs, dtype=torch.float32).unsqueeze(0)
    dj_t = torch.tensor(dj_norm, dtype=torch.float32).unsqueeze(0)

    net.eval()
    with torch.no_grad():
        net_preds = net(rv_t, rs_t, dj_t).squeeze(0).numpy()

    max_diff = float(np.max(np.abs(baseline_preds - net_preds)))
    assert max_diff < tol, (
        f"Equivalence check failed: max_diff = {max_diff:.2e} > tol = {tol:.2e}"
    )
    return max_diff


# ---------------------------------------------------------------------------
# Main window evaluation
# ---------------------------------------------------------------------------

def eval_window(
    rv_train: np.ndarray,
    rs_train: np.ndarray,
    dj_train: np.ndarray,
    rv_test: np.ndarray,
    rs_test: np.ndarray,
    dj_test: np.ndarray,
    horizons: tuple,
) -> dict:
    """
    Full pipeline for one index / window / architecture combination.

    Returns a dict with keys: baseline_mae/mse/qlike, harnet_mae/mse/qlike,
    rel_mae/mse/qlike, equiv_max_diff.
    """
    # --- Extended OLS baseline ---
    coef = fit_extended_ols(rv_train, rs_train, dj_train, horizons)
    baseline_preds_rv = extended_ols_test_preds(
        rv_train, rs_train, dj_train,
        rv_test, rs_test, dj_test,
        horizons, coef,
    )
    baseline_metrics = {m: LOSS_FN_NP[m](rv_test, baseline_preds_rv) for m in METRICS}

    # --- HARNetExtended ---
    net = HARNetExtended(horizons=horizons)
    net.init_from_extended_baseline(coef)

    equiv_diff = verify_equivalence(
        net, rv_train, rs_train, dj_train, coef, horizons
    )

    z_rv_train = scale_rv(rv_train)
    z_rs_train = scale_rv(rs_train)
    dj_norm_train = dj_train / RV_SCALE

    trainer = HARNetExtendedTrainer(
        net,
        loss_name="QLIKE",
        lr=LR,
        batch_size=BATCH_SIZE,
        seg_len=SEG_LEN,
        n_iter=N_ITER,
        seed=SEED,
    )
    trainer.train(z_rv_train, z_rs_train, dj_norm_train)

    rv_full = np.concatenate([rv_train, rv_test])
    rs_full = np.concatenate([rs_train, rs_test])
    dj_full = np.concatenate([dj_train, dj_test])

    z_rv_full = scale_rv(rv_full)
    z_rs_full = scale_rv(rs_full)
    dj_norm_full = dj_full / RV_SCALE

    test_preds_scaled = trainer.predict(
        z_rv_full, z_rs_full, dj_norm_full, n_train=len(rv_train)
    )
    clip_th = compute_clip_threshold(rv_train)
    test_preds_rv = clip_forecasts(unscale_rv(test_preds_scaled), clip_th)

    harnet_metrics = {m: LOSS_FN_NP[m](rv_test, test_preds_rv) for m in METRICS}

    result = {"equiv_max_diff": equiv_diff}
    for m in METRICS:
        result[f"baseline_{m.lower()}"] = baseline_metrics[m]
        result[f"harnet_{m.lower()}"] = harnet_metrics[m]
        b = baseline_metrics[m]
        result[f"rel_{m.lower()}"] = harnet_metrics[m] / b if b > 0 else np.nan

    return result


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    ext_data = load_extended_data()
    rows = []

    for index in INDEXES:
        df_index = ext_data[index]
        rv_series = df_index["rv"]
        rs_series = df_index["rs_neg"]
        dj_series = df_index["delta_j"]

        print(f"\n=== {index} ===")

        for win_id, (train_start_yr, train_end_yr, test_yr) in enumerate(WINDOWS):
            train_start, train_end, test_start, test_end = get_window_dates(
                train_start_yr, train_end_yr, test_yr
            )

            def _slice(s):
                return s.loc[train_start:train_end].values, s.loc[test_start:test_end].values

            rv_train, rv_test = _slice(rv_series)
            rs_train, rs_test = _slice(rs_series)
            dj_train, dj_test = _slice(dj_series)

            if len(rv_train) == 0 or len(rv_test) == 0:
                continue

            print(
                f"  Window {win_id + 1:02d} | test={test_yr} | "
                f"n_train={len(rv_train)}  n_test={len(rv_test)}"
            )

            for horizons in ARCH_LIST:
                horizons_str = str(horizons)
                try:
                    res = eval_window(
                        rv_train, rs_train, dj_train,
                        rv_test, rs_test, dj_test,
                        horizons,
                    )
                except AssertionError as e:
                    print(f"    WARNING — equivalence check: {e}")
                    continue

                row = {
                    "index": index,
                    "window_id": win_id + 1,
                    "test_yr": test_yr,
                    "horizons": horizons_str,
                    "equiv_max_diff": res["equiv_max_diff"],
                }
                for m in METRICS:
                    row[f"baseline_{m.lower()}"] = res[f"baseline_{m.lower()}"]
                    row[f"harnet_{m.lower()}"] = res[f"harnet_{m.lower()}"]
                    row[f"rel_{m.lower()}"] = res[f"rel_{m.lower()}"]
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_relative_errors(df: pd.DataFrame) -> None:
    """Boxplots of relative errors by architecture."""
    rel_cols = {m: f"rel_{m.lower()}" for m in METRICS}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))

    for ax, metric in zip(axes, METRICS):
        col = rel_cols[metric]
        data_har20 = df[df["horizons"] == str((1, 5, 20))][col].dropna().values
        data_har80 = df[df["horizons"] == str((1, 5, 20, 40, 80))][col].dropna().values
        bp = ax.boxplot(
            [data_har20, data_har80],
            tick_labels=["HAR20", "HAR80"],
            patch_artist=True,
        )
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(f"{metric}")
        ax.set_ylabel("HARNetExtended / Extended OLS")
        for patch, colour in zip(bp["boxes"], ["#1f77b4", "#ff7f0e"]):
            patch.set_facecolor(colour)
            patch.set_alpha(0.5)

    fig.suptitle("Relative errors: HARNetExtended vs Extended OLS Baseline", fontsize=10)
    fig.tight_layout()
    out_path = RESULTS_DIR / "exp5_relative_errors.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = run_experiment()

    csv_path = RESULTS_DIR / "exp5_extended_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_relative_errors(df)

    print("\n--- Median relative QLIKE by architecture ---")
    for horizons in ARCH_LIST:
        sub = df[df["horizons"] == str(horizons)]
        med = sub["rel_qlike"].median()
        print(f"  {horizons}: {med:.4f}")
