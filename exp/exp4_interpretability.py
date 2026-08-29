"""
Exp 4 — Interpretability: analysis of learned filter weights for SPX.

For each of the 15 rolling windows we:
  1. Fit an OLS HAR20 baseline on the training split.
  2. Initialise HARNet((1, 5, 20)) from the baseline coefficients.
  3. Train with QLIKE for 1 000 iterations.
  4. Call ``net.get_effective_daily_weights()`` to obtain the per-horizon
     effective lag weights.
  5. Normalise each feature's weights by the sum of absolute values.

Results are collected across all 15 windows and compared to the uniform
initialisation (1/5 per lag for weekly; 1/20 per lag for monthly).

Outputs
-------
  results/exp4_filter_weights.csv
  results/exp4_weekly_weights.png   (bar chart, lags 1-5)
  results/exp4_monthly_weights.png  (bar chart, lags 1-20)
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
from src.rolling_windows import WINDOWS, get_window_dates, split_series
from src.transforms import scale_rv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX = "SPX"
HORIZONS = (1, 5, 20)
N_ITER = 1000
LR = 1e-4
BATCH_SIZE = 4
SEG_LEN = 5
SEED = 42

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(weights: np.ndarray) -> np.ndarray:
    """Divide by sum of absolute values; return zeros if all-zero."""
    total = np.sum(np.abs(weights))
    return weights / total if total > 0 else weights


def extract_window_weights(
    rv_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit OLS HAR20, warm-start HARNet, train with QLIKE, return normalised
    effective lag weights for the weekly and monthly features.

    Returns
    -------
    weekly_weights : np.ndarray, shape (5,)
        Normalised weights for lags 1–5 (feature index 1).
    monthly_weights : np.ndarray, shape (20,)
        Normalised weights for lags 1–20 (feature index 2), sorted by lag.
    """
    baseline = HARBaseline("OLS", horizons=HORIZONS)
    baseline.fit(rv_train)

    net = HARNet(horizons=HORIZONS)
    net.init_from_baseline(baseline.coef_)

    z_train = scale_rv(rv_train)
    trainer = HARNetTrainer(
        net,
        loss_name="QLIKE",
        lr=LR,
        batch_size=BATCH_SIZE,
        seg_len=SEG_LEN,
        n_iter=N_ITER,
        seed=SEED,
    )
    trainer.train(z_train)

    eff_weights = net.get_effective_daily_weights()

    # Feature index 1 = weekly (lags 1-5)
    _, w_weekly = eff_weights[1]
    weekly_norm = _normalise(w_weekly)

    # Feature index 2 = monthly (lags 1-20 composed)
    lags_monthly, w_monthly = eff_weights[2]
    # Sort by lag to ensure consistent ordering
    order = np.argsort(lags_monthly)
    monthly_norm = _normalise(w_monthly[order])

    return weekly_norm, monthly_norm


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    rv_data = load_oxford_man_data()
    rv_series = rv_data[INDEX]

    rows = []
    print(f"Running Exp4 (interpretability) for {INDEX} …")

    for win_id, (train_start_yr, train_end_yr, test_yr) in enumerate(WINDOWS):
        train_start, train_end, test_start, test_end = get_window_dates(
            train_start_yr, train_end_yr, test_yr
        )
        rv_train, _ = split_series(
            rv_series, train_start, train_end, test_start, test_end
        )
        rv_train_np = rv_train.values
        if len(rv_train_np) < max(HORIZONS) + SEG_LEN:
            continue

        print(f"  Window {win_id + 1:02d} | test={test_yr} | n_train={len(rv_train_np)}")

        weekly_w, monthly_w = extract_window_weights(rv_train_np)

        for lag_idx, w in enumerate(weekly_w, start=1):
            rows.append(
                {
                    "window_id": win_id + 1,
                    "feature": "weekly",
                    "lag": lag_idx,
                    "weight": float(w),
                }
            )
        for lag_idx, w in enumerate(monthly_w, start=1):
            rows.append(
                {
                    "window_id": win_id + 1,
                    "feature": "monthly",
                    "lag": lag_idx,
                    "weight": float(w),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_weekly_weights(df: pd.DataFrame) -> None:
    """Bar chart of median (±IQR) weekly weights over lags 1-5."""
    sub = df[df["feature"] == "weekly"]
    lags = sorted(sub["lag"].unique())
    medians = [sub[sub["lag"] == lag]["weight"].median() for lag in lags]
    q25 = [sub[sub["lag"] == lag]["weight"].quantile(0.25) for lag in lags]
    q75 = [sub[sub["lag"] == lag]["weight"].quantile(0.75) for lag in lags]
    err_lo = [m - lo for m, lo in zip(medians, q25)]
    err_hi = [hi - m for m, hi in zip(medians, q75)]

    uniform_val = 1.0 / len(lags)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(lags))
    ax.bar(x, medians, yerr=[err_lo, err_hi], capsize=4, color="#1f77b4", alpha=0.7,
           label="Learned (median ± IQR)")
    ax.axhline(uniform_val, color="r", linestyle="--", linewidth=1.2,
               label=f"Uniform init (1/{len(lags)})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Lag {lag}" for lag in lags])
    ax.set_xlabel("Daily lag")
    ax.set_ylabel("Normalised weight")
    ax.set_title(f"SPX — Weekly feature effective weights (15 windows)")
    ax.legend()
    fig.tight_layout()
    out_path = RESULTS_DIR / "exp4_weekly_weights.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_monthly_weights(df: pd.DataFrame) -> None:
    """Bar chart of median effective monthly weights by lag (1-20)."""
    sub = df[df["feature"] == "monthly"]
    lags = sorted(sub["lag"].unique())
    medians = [sub[sub["lag"] == lag]["weight"].median() for lag in lags]
    q25 = [sub[sub["lag"] == lag]["weight"].quantile(0.25) for lag in lags]
    q75 = [sub[sub["lag"] == lag]["weight"].quantile(0.75) for lag in lags]
    err_lo = [m - lo for m, lo in zip(medians, q25)]
    err_hi = [hi - m for m, hi in zip(medians, q75)]

    uniform_val = 1.0 / len(lags)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(lags))
    ax.bar(x, medians, yerr=[err_lo, err_hi], capsize=2, color="#2ca02c", alpha=0.7,
           label="Learned (median ± IQR)")
    ax.axhline(uniform_val, color="r", linestyle="--", linewidth=1.2,
               label=f"Uniform init (1/{len(lags)})")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{lags[i]}" for i in range(0, len(lags), 2)])
    ax.set_xlabel("Daily lag")
    ax.set_ylabel("Normalised weight")
    ax.set_title(f"SPX — Monthly feature effective weights (15 windows)")
    ax.legend()
    fig.tight_layout()
    out_path = RESULTS_DIR / "exp4_monthly_weights.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = run_experiment()

    csv_path = RESULTS_DIR / "exp4_filter_weights.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    plot_weekly_weights(df)
    plot_monthly_weights(df)

    # Quick summary
    print("\n--- Median weekly weights across 15 windows ---")
    sub = df[df["feature"] == "weekly"]
    for lag in sorted(sub["lag"].unique()):
        med = sub[sub["lag"] == lag]["weight"].median()
        print(f"  Lag {lag}: {med:.4f}  (uniform = {1/5:.4f})")

    print("\n--- Median monthly weights (first 5 lags) ---")
    sub = df[df["feature"] == "monthly"]
    for lag in sorted(sub["lag"].unique())[:5]:
        med = sub[sub["lag"] == lag]["weight"].median()
        print(f"  Lag {lag:2d}: {med:.4f}  (uniform = {1/20:.4f})")
