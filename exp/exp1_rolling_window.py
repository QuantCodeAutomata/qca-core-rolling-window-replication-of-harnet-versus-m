"""
Experiment 1 — Rolling-window HARNet vs HAR baselines.

Iterates over:
  Indexes    : SPX, FTSE, DJI
  Windows    : 15 rolling windows (4-year train, 1-year test, 2006–2020)
  Architectures: HAR20 = (1,5,20), HAR80 = (1,5,20,40,80)
  Fit types  : OLS, WLS, log-OLS
  Losses     : MAE, MSE, QLIKE

NOTE: n_iter=1000 (paper uses 10000 but reduced here for speed).

Outputs
-------
results/exp1_results.csv       — raw per-window metrics
results/exp1_boxplots_har20.png — 3×3 relative-QLIKE boxplots (HAR20)
results/exp1_boxplots_har80.png — 3×3 relative-QLIKE boxplots (HAR80)
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
from src.experiment_runner import run_window_config
from src.rolling_windows import WINDOWS

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

INDEXES = ["SPX", "FTSE", "DJI"]

ARCHS = {
    "HAR20": (1, 5, 20),
    "HAR80": (1, 5, 20, 40, 80),
}

FIT_TYPES = ["OLS", "WLS", "log-OLS"]
LOSS_NAMES = ["MAE", "MSE", "QLIKE"]

N_ITER = 1000  # paper uses 10000; reduced for speed

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _make_boxplot_figure(df_arch: pd.DataFrame, arch_name: str) -> plt.Figure:
    """
    3×3 grid of boxplots — rows=loss_name, cols=fit_type, y=relative_qlike.

    Each cell shows the distribution across all windows × indexes
    (15 windows × 3 indexes = 45 data points).
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=False)
    fig.suptitle(
        f"Exp1 — Relative QLIKE (HARNet / baseline) — {arch_name}",
        fontsize=13,
        fontweight="bold",
    )

    for i, loss_name in enumerate(LOSS_NAMES):
        for j, fit_type in enumerate(FIT_TYPES):
            ax = axes[i, j]
            mask = (df_arch["loss_name"] == loss_name) & (df_arch["fit_type"] == fit_type)
            values = df_arch.loc[mask, "relative_qlike"].values

            ax.boxplot(values, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.6),
                       medianprops=dict(color="navy", linewidth=2))
            ax.axhline(1.0, color="crimson", linestyle="--", linewidth=0.9)
            ax.set_title(f"loss={loss_name} | fit={fit_type}", fontsize=8)
            ax.set_xticks([])
            if j == 0:
                ax.set_ylabel("rel. QLIKE", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(
        f"NOTE: n_iter={N_ITER} (paper uses 10000 but reduced for speed).\n"
        f"      Total configurations: "
        f"{len(INDEXES)} indexes × {len(WINDOWS)} windows × "
        f"{len(ARCHS)} archs × {len(FIT_TYPES)} fit_types × {len(LOSS_NAMES)} losses "
        f"= {len(INDEXES)*len(WINDOWS)*len(ARCHS)*len(FIT_TYPES)*len(LOSS_NAMES)}"
    )

    data = load_oxford_man_data()

    records: list[dict] = []
    total = len(INDEXES) * len(WINDOWS) * len(ARCHS) * len(FIT_TYPES) * len(LOSS_NAMES)
    count = 0

    for index in INDEXES:
        series = data[index]

        for train_start_yr, train_end_yr, test_yr in WINDOWS:
            train = series[
                (series.index.year >= train_start_yr)
                & (series.index.year <= train_end_yr)
            ]
            test = series[series.index.year == test_yr]

            for arch_name, horizons in ARCHS.items():
                for fit_type in FIT_TYPES:
                    for loss_name in LOSS_NAMES:
                        count += 1
                        print(
                            f"[{count:4d}/{total}] "
                            f"idx={index}  win={train_start_yr}-{train_end_yr}/{test_yr}  "
                            f"arch={arch_name}  fit={fit_type}  loss={loss_name}"
                        )

                        result = run_window_config(
                            rv_train=train.values,
                            rv_test=test.values,
                            horizons=horizons,
                            fit_type=fit_type,
                            loss_name=loss_name,
                            n_iter=N_ITER,
                            seed=42,
                        )

                        records.append(
                            {
                                "index": index,
                                "train_start_yr": train_start_yr,
                                "train_end_yr": train_end_yr,
                                "test_yr": test_yr,
                                "horizons": str(horizons),
                                "fit_type": fit_type,
                                "loss_name": loss_name,
                                "baseline_mae": result["baseline_mae"],
                                "baseline_mse": result["baseline_mse"],
                                "baseline_qlike": result["baseline_qlike"],
                                "harnet_mae": result["harnet_mae"],
                                "harnet_mse": result["harnet_mse"],
                                "harnet_qlike": result["harnet_qlike"],
                                "relative_mae": result["relative_mae"],
                                "relative_mse": result["relative_mse"],
                                "relative_qlike": result["relative_qlike"],
                                "pre_train_max_diff": result["pre_train_max_diff"],
                            }
                        )

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "exp1_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # -----------------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------------
    print("\nMedian relative_qlike by (horizons, fit_type, loss_name):")
    summary = (
        df.groupby(["horizons", "fit_type", "loss_name"])["relative_qlike"]
        .median()
        .rename("median_rel_qlike")
    )
    print(summary.to_string())

    # -----------------------------------------------------------------------
    # Boxplots — one figure per architecture
    # -----------------------------------------------------------------------
    for arch_name, horizons in ARCHS.items():
        horizons_str = str(horizons)
        df_arch = df[df["horizons"] == horizons_str].copy()

        fig = _make_boxplot_figure(df_arch, arch_name)
        png_path = RESULTS_DIR / f"exp1_boxplots_{arch_name.lower()}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Boxplot saved to {png_path}")

    print("Exp1 complete. Results saved to results/exp1_results.csv")


if __name__ == "__main__":
    main()
