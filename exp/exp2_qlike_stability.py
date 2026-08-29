"""
Exp 2 — QLIKE optimisation stability: HAR-baseline init vs Glorot random init.

For three fixed training windows (one per index) we run 10 independent HAR20
training trials under each initialisation strategy and record how the QLIKE
training loss evolves.

Fixed windows
-------------
  SPX  : train 2006-01-01 – 2009-12-31
  FTSE : train 2013-01-01 – 2016-12-31
  DJI  : train 2003-01-01 – 2006-12-31

Expected paper values (medians)
--------------------------------
  SPX  random ~5.62,  HAR-init ~0.161
  FTSE random ~27.08, HAR-init ~0.256
  DJI  random ~1.54,  HAR-init ~0.127

Outputs
-------
  results/exp2_stability.csv
  results/exp2_stability_trajectories.png
  results/exp2_stability_boxplots.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.init as nn_init

from data.generate_data import load_oxford_man_data
from src.har_model import HARBaseline
from src.harnet import HARNet, HARNetTrainer
from src.losses import LOSS_FN_TORCH, qlike_loss_torch
from src.transforms import scale_rv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZONS = (1, 5, 20)
LOSS_NAME = "QLIKE"
N_RUNS = 10
N_ITER = 1000
LR = 1e-4
BATCH_SIZE = 4
SEG_LEN = 5

# Fixed training windows: (index, train_start, train_end)
FIXED_WINDOWS = [
    ("SPX",  "2006-01-01", "2009-12-31"),
    ("FTSE", "2013-01-01", "2016-12-31"),
    ("DJI",  "2003-01-01", "2006-12-31"),
]

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def glorot_init_harnet(net: HARNet) -> None:
    """Apply Glorot uniform initialisation to all parameters of *net*."""
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                nn_init.xavier_uniform_(p)
            else:
                nn_init.uniform_(p, -0.1, 0.1)


def compute_full_loss(net: HARNet, z_train: np.ndarray) -> float:
    """Compute QLIKE on the entire training series (no gradient)."""
    net.eval()
    z_t = torch.tensor(z_train, dtype=torch.float32).unsqueeze(0)
    burn_in = net.horizons[-1]
    with torch.no_grad():
        preds = net(z_t)                                    # (1, T-burn_in)
        targets = z_t[:, burn_in:]
        loss = qlike_loss_torch(targets, preds)
    return float(loss.item())


def run_single(
    z_train: np.ndarray,
    init_type: str,
    har_coef: np.ndarray,
    seed: int,
) -> dict:
    """
    One training run.

    Returns a dict with keys: init_loss, final_loss, losses (full trajectory).
    """
    net = HARNet(horizons=HORIZONS)

    if init_type == "har":
        net.init_from_baseline(har_coef)
    else:
        torch.manual_seed(seed)
        glorot_init_harnet(net)

    init_loss = compute_full_loss(net, z_train)

    trainer = HARNetTrainer(
        net,
        loss_name=LOSS_NAME,
        lr=LR,
        batch_size=BATCH_SIZE,
        seg_len=SEG_LEN,
        n_iter=N_ITER,
        seed=seed,
    )
    losses = trainer.train(z_train)
    final_loss = losses[-1]

    return {"init_loss": init_loss, "final_loss": final_loss, "losses": losses}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    rv_data = load_oxford_man_data()
    records = []

    for index, train_start, train_end in FIXED_WINDOWS:
        rv_series = rv_data[index]
        rv_train = rv_series.loc[train_start:train_end].values
        z_train = scale_rv(rv_train)

        # Fit OLS baseline once to get HAR init coefficients
        baseline = HARBaseline("OLS", horizons=HORIZONS)
        baseline.fit(rv_train)
        har_coef = baseline.coef_

        window_label = f"{index} {train_start[:4]}–{train_end[:4]}"
        print(f"\n=== {window_label} ===")

        all_losses: dict[str, list[list[float]]] = {"har": [], "random": []}

        for seed in range(N_RUNS):
            for init_type in ("har", "random"):
                result = run_single(z_train, init_type, har_coef, seed)
                records.append(
                    {
                        "index": index,
                        "window": window_label,
                        "init_type": init_type,
                        "seed": seed,
                        "init_loss": result["init_loss"],
                        "final_loss": result["final_loss"],
                    }
                )
                all_losses[init_type].append(result["losses"])

        # Print summary
        for init_type in ("har", "random"):
            finals = [r["final_loss"] for r in records
                      if r["index"] == index and r["init_type"] == init_type]
            label = "HAR-init" if init_type == "har" else "Glorot  "
            print(
                f"  {label}: median={np.median(finals):.4f}  "
                f"mean={np.mean(finals):.4f}  std={np.std(finals):.4f}"
            )

        _plot_trajectories(all_losses, window_label)

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_trajectories(
    all_losses: dict[str, list[list[float]]],
    window_label: str,
) -> None:
    """Store trajectory data for later combined plotting (not standalone)."""
    # Defer to the combined trajectory plot in main()
    pass


def plot_trajectories(df_records: pd.DataFrame, all_traj: dict) -> None:
    """Loss-curve panel: one subplot per fixed window."""
    windows = list(dict.fromkeys(df_records["window"].tolist()))
    fig, axes = plt.subplots(1, len(windows), figsize=(5 * len(windows), 4), sharey=False)
    if len(windows) == 1:
        axes = [axes]

    colours = {"har": "#1f77b4", "random": "#d62728"}
    labels = {"har": "HAR-init", "random": "Glorot random"}

    for ax, win in zip(axes, windows):
        for init_type in ("har", "random"):
            trajectories = all_traj[win][init_type]
            iters = np.arange(1, len(trajectories[0]) + 1)
            mat = np.array(trajectories)
            median = np.median(mat, axis=0)
            lo = np.percentile(mat, 25, axis=0)
            hi = np.percentile(mat, 75, axis=0)
            ax.semilogy(iters, median, color=colours[init_type], label=labels[init_type])
            ax.fill_between(iters, lo, hi, alpha=0.2, color=colours[init_type])
        ax.set_title(win, fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("QLIKE (log scale)")
        ax.legend(fontsize=8)

    fig.suptitle("QLIKE Training Loss: HAR-init vs Glorot Random (10 seeds)", fontsize=10)
    fig.tight_layout()
    out_path = RESULTS_DIR / "exp2_stability_trajectories.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_boxplots(df: pd.DataFrame) -> None:
    """Boxplot of final QLIKE by init type, one subplot per fixed window."""
    windows = list(dict.fromkeys(df["window"].tolist()))
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4), sharey=False)
    if len(windows) == 1:
        axes = [axes]

    for ax, win in zip(axes, windows):
        sub = df[df["window"] == win]
        data_har = sub[sub["init_type"] == "har"]["final_loss"].values
        data_rnd = sub[sub["init_type"] == "random"]["final_loss"].values
        ax.boxplot(
            [data_har, data_rnd],
            tick_labels=["HAR-init", "Glorot"],
            patch_artist=True,
            boxprops=dict(facecolor="#1f77b4", alpha=0.5),
        )
        ax.set_title(win, fontsize=9)
        ax.set_ylabel("Final QLIKE")
        ax.set_yscale("log")

    fig.suptitle("Final QLIKE Distribution after 1000 Iterations", fontsize=10)
    fig.tight_layout()
    out_path = RESULTS_DIR / "exp2_stability_boxplots.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rv_data = load_oxford_man_data()

    # Collect full trajectories for the trajectory plot
    all_traj: dict[str, dict[str, list[list[float]]]] = {}
    records = []

    for index, train_start, train_end in FIXED_WINDOWS:
        rv_series = rv_data[index]
        rv_train = rv_series.loc[train_start:train_end].values
        z_train = scale_rv(rv_train)

        baseline = HARBaseline("OLS", horizons=HORIZONS)
        baseline.fit(rv_train)
        har_coef = baseline.coef_

        window_label = f"{index} {train_start[:4]}–{train_end[:4]}"
        all_traj[window_label] = {"har": [], "random": []}
        print(f"\n=== {window_label} ===")

        for seed in range(N_RUNS):
            for init_type in ("har", "random"):
                result = run_single(z_train, init_type, har_coef, seed)
                records.append(
                    {
                        "index": index,
                        "window": window_label,
                        "init_type": init_type,
                        "seed": seed,
                        "init_loss": result["init_loss"],
                        "final_loss": result["final_loss"],
                    }
                )
                all_traj[window_label][init_type].append(result["losses"])

        # Print summary
        for init_type in ("har", "random"):
            finals = [
                r["final_loss"]
                for r in records
                if r["index"] == index and r["init_type"] == init_type
            ]
            label = "HAR-init" if init_type == "har" else "Glorot  "
            print(
                f"  {label}: median={np.median(finals):.4f}  "
                f"mean={np.mean(finals):.4f}  std={np.std(finals):.4f}"
            )

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "exp2_stability.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_trajectories(df, all_traj)
    plot_boxplots(df)

    print("\nExpected paper medians (approx.):")
    print("  SPX  random ~5.62,  HAR-init ~0.161")
    print("  FTSE random ~27.08, HAR-init ~0.256")
    print("  DJI  random ~1.54,  HAR-init ~0.127")
