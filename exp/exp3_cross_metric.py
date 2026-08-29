"""
Exp 3 — Cross-metric evaluation: preferred HARNet (OLS init + QLIKE) vs baselines.

For each of the 3 indexes and 15 rolling windows the preferred HARNet is trained
with QLIKE loss (OLS init) using both HAR20 and HAR80.  The test-set performance
is reported for three metrics (MAE, MSE, QLIKE) and compared to three linear
baselines (OLS, WLS, log-OLS).

Relative ratios ``preferred_HARNet / baseline`` are stored alongside the
ex-post best baseline (minimum metric across OLS/WLS/log-OLS per window).

Outputs
-------
  results/exp3_cross_metric.csv
  results/exp3_relative_errors.png
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

from data.generate_data import load_oxford_man_data
from src.har_model import HARBaseline
from src.harnet import HARNet, HARNetTrainer
from src.losses import LOSS_FN_NP
from src.rolling_windows import WINDOWS, get_window_dates, split_series
from src.transforms import (
    clip_forecasts,
    compute_clip_threshold,
    scale_rv,
    unscale_rv,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEXES = ["SPX", "FTSE", "DJI"]
ARCH_LIST = [(1, 5, 20), (1, 5, 20, 40, 80)]
METRICS = ["MAE", "MSE", "QLIKE"]
N_ITER = 1000
LR = 1e-4
BATCH_SIZE = 4
SEG_LEN = 5

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eval_baseline_only(
    rv_train: np.ndarray,
    rv_test: np.ndarray,
    horizons: tuple,
    fit_type: str,
) -> dict:
    """
    Fit a linear HAR baseline and compute out-of-sample metrics.

    Parameters
    ----------
    rv_train, rv_test : np.ndarray
        Raw RV training and test series.
    horizons : tuple of int
    fit_type : str
        One of ``'OLS'``, ``'WLS'``, ``'log-OLS'``.

    Returns
    -------
    dict
        Keys ``'MAE'``, ``'MSE'``, ``'QLIKE'`` (RV-scale metrics).
    """
    model = HARBaseline(fit_type=fit_type, horizons=horizons)
    model.fit(rv_train)

    rv_full = np.concatenate([rv_train, rv_test])
    burn_in = max(horizons)
    n_train = len(rv_train)

    # predict_rv_scale returns predictions for all valid positions (burn_in … end)
    preds_rv = model.predict_rv_scale(rv_full)
    test_preds = preds_rv[n_train - burn_in :]   # align to test period

    clip_th = compute_clip_threshold(rv_train)
    test_preds = clip_forecasts(test_preds, clip_th)

    return {m: LOSS_FN_NP[m](rv_test, test_preds) for m in METRICS}


def eval_harnet(
    rv_train: np.ndarray,
    rv_test: np.ndarray,
    horizons: tuple,
    seed: int = 0,
) -> dict:
    """
    Train preferred HARNet (OLS init + QLIKE) and compute out-of-sample metrics.

    Returns
    -------
    dict
        Keys ``'MAE'``, ``'MSE'``, ``'QLIKE'`` (RV-scale metrics).
    """
    baseline = HARBaseline("OLS", horizons=horizons)
    baseline.fit(rv_train)

    net = HARNet(horizons=horizons)
    net.init_from_baseline(baseline.coef_)

    z_train = scale_rv(rv_train)
    trainer = HARNetTrainer(
        net,
        loss_name="QLIKE",
        lr=LR,
        batch_size=BATCH_SIZE,
        seg_len=SEG_LEN,
        n_iter=N_ITER,
        seed=seed,
    )
    trainer.train(z_train)

    rv_full = np.concatenate([rv_train, rv_test])
    z_full = scale_rv(rv_full)
    n_train = len(rv_train)

    test_preds_scaled = trainer.predict(z_full, start_idx=n_train)
    clip_th = compute_clip_threshold(rv_train)
    test_preds = clip_forecasts(unscale_rv(test_preds_scaled), clip_th)

    return {m: LOSS_FN_NP[m](rv_test, test_preds) for m in METRICS}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    rv_data = load_oxford_man_data()
    rows = []

    for index in INDEXES:
        rv_series = rv_data[index]
        print(f"\n=== {index} ===")

        for win_id, (train_start_yr, train_end_yr, test_yr) in enumerate(WINDOWS):
            train_start, train_end, test_start, test_end = get_window_dates(
                train_start_yr, train_end_yr, test_yr
            )
            rv_train, rv_test = split_series(
                rv_series, train_start, train_end, test_start, test_end
            )
            rv_train = rv_train.values
            rv_test = rv_test.values

            if len(rv_train) == 0 or len(rv_test) == 0:
                continue

            print(f"  Window {win_id + 1:02d} | test={test_yr} | "
                  f"n_train={len(rv_train)}  n_test={len(rv_test)}")

            for horizons in ARCH_LIST:
                horizons_str = str(horizons)

                # Preferred HARNet
                harnet_metrics = eval_harnet(rv_train, rv_test, horizons)

                # Three linear baselines
                ols_metrics = eval_baseline_only(rv_train, rv_test, horizons, "OLS")
                wls_metrics = eval_baseline_only(rv_train, rv_test, horizons, "WLS")
                logols_metrics = eval_baseline_only(rv_train, rv_test, horizons, "log-OLS")

                for metric in METRICS:
                    hn = harnet_metrics[metric]
                    ols = ols_metrics[metric]
                    wls = wls_metrics[metric]
                    logols = logols_metrics[metric]
                    ex_post = min(ols, wls, logols)

                    rows.append(
                        {
                            "index": index,
                            "window_id": win_id + 1,
                            "test_yr": test_yr,
                            "horizons": horizons_str,
                            "metric": metric,
                            "preferred_harnet": hn,
                            "ols_baseline": ols,
                            "wls_baseline": wls,
                            "logols_baseline": logols,
                            "ex_post_best": ex_post,
                            "rel_vs_ols": hn / ols if ols > 0 else np.nan,
                            "rel_vs_wls": hn / wls if wls > 0 else np.nan,
                            "rel_vs_logols": hn / logols if logols > 0 else np.nan,
                            "rel_vs_expost": hn / ex_post if ex_post > 0 else np.nan,
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_relative_errors(df: pd.DataFrame) -> None:
    """Boxplots of relative ratios vs each baseline for HAR20."""
    har20_df = df[df["horizons"] == str((1, 5, 20))]

    rel_cols = ["rel_vs_ols", "rel_vs_wls", "rel_vs_logols", "rel_vs_expost"]
    labels = ["vs OLS", "vs WLS", "vs log-OLS", "vs Ex-post best"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4), sharey=False)

    for ax, metric in zip(axes, METRICS):
        sub = har20_df[har20_df["metric"] == metric]
        data = [sub[col].dropna().values for col in rel_cols]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(f"HAR20 — {metric}", fontsize=9)
        ax.set_ylabel("HARNet / Baseline")
        ax.tick_params(axis="x", labelrotation=15)
        for patch in bp["boxes"]:
            patch.set_facecolor("#1f77b4")
            patch.set_alpha(0.5)

    fig.suptitle("Relative errors: Preferred HARNet (OLS+QLIKE) vs Baselines", fontsize=10)
    fig.tight_layout()
    out_path = RESULTS_DIR / "exp3_relative_errors.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = run_experiment()

    csv_path = RESULTS_DIR / "exp3_cross_metric.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_relative_errors(df)

    # Quick summary
    print("\n--- Median relative ratios vs OLS (HAR20, QLIKE metric) ---")
    sub = df[(df["horizons"] == str((1, 5, 20))) & (df["metric"] == "QLIKE")]
    print(f"  rel_vs_ols    median: {sub['rel_vs_ols'].median():.4f}")
    print(f"  rel_vs_wls    median: {sub['rel_vs_wls'].median():.4f}")
    print(f"  rel_vs_logols median: {sub['rel_vs_logols'].median():.4f}")
    print(f"  rel_vs_expost median: {sub['rel_vs_expost'].median():.4f}")
