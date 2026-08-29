# AGENTS.md — HARNet Replication Study

## Repository Overview

Replication study of the HARNet model for **daily realized volatility
forecasting**. HARNet is a convolutional neural network that mirrors the
Heterogeneous AutoRegressive (HAR) lag structure while learning non-linear
transformations of the input. The codebase provides:

- HAR linear baselines (OLS, WLS, log-OLS)
- A custom PyTorch HARNet (two variants: HAR20 and HAR80)
- A rolling-window evaluation framework (15 windows, 3 indexes, 3 losses)
- Five experiments reproducing key results from the original paper

---

## Data

**Source:** Synthetic realized variance data mimicking the
[Oxford-Man Institute Realized Library](https://realized.oxford-man.ox.ac.uk/)
format. The real library is not accessible in this environment.

**Indexes:** SPX, FTSE, DJI

**Period:** Business days 2001-01-02 through 2020-12-31

**Generator:** `data/generate_data.py` — produces a DataFrame with a
`DatetimeIndex` and one RV column per index, saved to `data/rv_data.parquet`.

---

## Key Modules

| Module | Purpose |
|---|---|
| `src/harnet.py` | Custom PyTorch HARNet; `HARNet` class with configurable horizons and loss |
| `src/harnet_extended.py` | Three-channel HARNetExtended (RV + RS- + ΔJ) for Exp5 |
| `src/har_model.py` | OLS, WLS, and log-OLS HAR baselines via `statsmodels` |
| `src/transforms.py` | RV scaling helpers (level and log-RV normalisation) |
| `src/losses.py` | `mae_loss`, `mse_loss`, `qlike_loss` — all accept raw (un-logged) predictions |
| `src/metrics.py` | Out-of-sample MAE, MSE, QLIKE computation with clipping |
| `data/generate_data.py` | Synthetic RV + semivariance data (Oxford-Man mimic) |
| `data/generate_extended_data.py` | Returns `{ticker: DataFrame[rv, rs_neg, delta_j]}` with seed=2024 noise |

---

## Architecture

### HAR20
- **Horizons:** 1, 5, 20 days
- **Parameters:** 13 (3 lag aggregators × 4 conv weights + 1 bias)
- **Input length:** 20 time steps

### HAR80
- **Horizons:** 1, 5, 20, 40, 80 days
- **Parameters:** 19
- **Input length:** 80 time steps

Both variants use a single convolutional layer per horizon, ReLU activations, and
a linear output layer that combines all horizon features.

---

## Scaling Rules

| Mode | Formula |
|---|---|
| Level RV | `rv_scaled = 1000 * rv` |
| Log-RV | `log_rv_scaled = (log(rv) + 13) / 10.5` |

Scaling constants are derived from the synthetic dataset statistics to keep
network inputs in a numerically stable range. Apply the inverse transform when
converting predictions back to raw RV before computing metrics.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | ADAM |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Segment length (`seg_len`) | 5 |
| Iterations | 10 000 |
| Early stopping | None |
| Loss functions | MAE, MSE, QLIKE |

Training uses segment-sampled mini-batches: each batch contains `batch_size`
randomly drawn contiguous segments of length `seg_len` from the training split.

---

## Clipping Rule

Before computing any out-of-sample metric, raw predictions are clipped to a
minimum of **0.5 × min(train RV)**:

```python
pred_clipped = np.maximum(pred, 0.5 * train_rv.min())
```

This prevents QLIKE from diverging when predictions are near or below zero.

---

## Rolling Windows

| Parameter | Value |
|---|---|
| Number of windows | 15 |
| Training length | 4 years |
| Test length | 1 year |
| First window | train 2002–2005, test 2006 |
| Last window | train 2016–2019, test 2020 |

---

## Running Experiments

All commands should be run from the repository root:

```bash
cd /workspace/project

# Experiment 1 — rolling-window HARNet vs HAR baselines
python exp/exp1_rolling_window.py

# Experiment 2 — QLIKE optimisation stability (HAR-init vs random init)
python exp/exp2_qlike_stability.py

# Experiment 3 — cross-metric evaluation of preferred HARNet (OLS + QLIKE)
python exp/exp3_cross_metric.py

# Experiment 4 — interpretability of learned HARNet filters
python exp/exp4_interpretability.py

# Experiment 5 — extended-input HARNet (RS- and signed jump variation)
python exp/exp5_extended_input.py
```

Outputs (CSV tables and PNG/PDF figures) are written to `results/`.

---

## Running Tests

```bash
cd /workspace/project && pytest tests/ -v
```

Tests cover: scaling/inverse-scaling round-trips, loss function shapes and
numerical properties, HAR feature construction, HARNet forward pass shapes, and
rolling-window index generation.

---

## Implementation Notes (Added During Development)

### Data Generation (`data/generate_data.py`)
- HAR DGP: `log(RV_t) = -1.4 + 0.3*daily + 0.3*weekly + 0.2*monthly + N(0, 0.09)`
- Constant `-1.4` chosen so stationary mean = `-1.4/0.2 = -7.0` → RV ≈ exp(-7) ≈ 9e-4
- 500-day warm-up: log(RV) initialised at stationary mean, then iterated forward
- Seeds: SPX=42, FTSE=43, DJI=44; unknown tickers use ASCII-code sum
- `rv_minus = clip(0.40*RV + N(0, (0.05*RV)²), 0.01*RV, 0.99*RV)` → fraction ≈ 0.40
- `signed_jump = RV - 2*rv_minus` (= RS+ − RS−)

### HARNet Forward Pass (`src/harnet.py`)
- Causal padding: `F.pad(current, (fl * dil, 0))` prepends `fl*dil` zeros
- Dilated tap gathering (vectorised, no time-step loop):
  ```python
  taps = torch.stack(
      [current_pad[:, (fl-1-j)*dil : (fl-1-j)*dil + T] for j in range(fl)],
      dim=2
  )  # (batch, T, fl)
  ```
  Tap j gives `current[t - (j+1)*dil]` at each position t (j=0 = closest lag)
- All hidden features computed over all T positions; only positions `[burn_in:T]`
  are used for prediction (the daily x1 uses `z[:, burn_in-1:T-1]`)
- Parameter counts: HAR20 = 5+4 filters + 3+1 output = **13**; HAR80 = 5+4+2+2 + 5+1 = **19**

### HAR Baseline (`src/har_model.py`)
- `log-OLS` stores residuals in the **scaled-log** space (`scale_log_rv` units)
- `predict_rv_scale` for `log-OLS` converts residuals back to natural log space
  via `residuals_log = residuals_ * LOG_RANGE` before calling
  `proietti_lutkepohl_bias_correction`
- WLS uses linearly increasing weights `w_i = (i+1) / sum(1..n)` via `sm.WLS`

### Effective Lag Weights (`get_effective_daily_weights`)
- Computed by linear composition of filter layers (ReLU treated as identity)
- HAR20 daily x1: lags=[1], HAR20 weekly h2: lags=1..5, HAR20 monthly h3: lags=1..20
  (layer k≥1 uses delay 0, dil, 2*dil, ... so monthly covers 4×5 = lags 1-20 exactly)
- With uniform init all 20 monthly weights = 1/20; sum of all weights per feature = 1.0
- Returned as list of `(lags_array, weights_array)` tuples, one per horizon

### Experiment Runner (`src/experiment_runner.py`)
- `run_window_config`: full pipeline — scale → fit baseline → init HARNet → check
  parity → train → forecast on full series → unscale → clip → metrics
- Pre-train parity check: `baseline.predict(rv_train)` (z-scale, no start_idx) vs
  `trainer.predict(z_train, start_idx=burn_in)` — both return shape
  `(len(rv_train) - burn_in,)`; assert max abs diff < 1e-5 (observed ~1e-7 in practice)
- Test forecasts: `baseline.predict(rv_full, start_idx=len(rv_train))` takes raw rv
  (internally transforms), returns exactly `len(rv_test)` z-scale predictions;
  `trainer.predict(z_full, start_idx=len(rv_train))` takes z-scaled full series
- log-OLS back-transform for HARNet: reuses `baseline.residuals_ * LOG_RANGE` as the
  log-space residuals for the Proietti-Lütkepohl σ² estimate (HARNet has no own residuals)
- `glorot_init_harnet`: 1-D params (filter weights, bias) handled via unsqueeze(0) →
  xavier_uniform_ → copy_ squeeze; 2-D params handled directly
- `compute_all_test_metrics`: thin wrapper over `LOSS_FN_NP` dict → {'MAE','MSE','QLIKE'}

### Exp1 Rolling Window (`exp/exp1_rolling_window.py`)
- Loop order: index → window → arch (HAR20/HAR80) → fit_type → loss_name
- Train/test split: `series[(series.index.year >= start) & (series.index.year <= end)]`
- `horizons` stored as `str(tuple)` in CSV, e.g. `'(1, 5, 20)'` for HAR20 filtering
- Boxplots: 3×3 grid, rows=LOSS_NAMES, cols=FIT_TYPES, y=relative_qlike; red dashed
  baseline at 1.0; one figure per arch saved to `results/exp1_boxplots_{har20,har80}.png`
- `matplotlib.use('Agg')` set before any pyplot import (required for headless environments)
- `sys.path.insert(0, str(Path(__file__).parent.parent))` placed after stdlib imports but
  before all project imports (all at top of file as per style rules)

### HARNetExtended (`src/harnet_extended.py`)
- Three-channel: rv_filters + rs_filters (same dilated pyramid as HARNet each) + ΔJ linear input
- Output layer: `Linear(2*n_horizons+1, 1, bias=True)`
- Parameter counts: HAR20=26 (2×9 filters + 7 weights + 1 bias), HAR80=38
- ΔJ input convention: pass `delta_j / RV_SCALE` (already normalised) as third arg to forward
  Forward slices ΔJ at `burn_in-1:T-1` (same causal lag as the daily RV feature)
- `init_from_extended_baseline(coef)`: coef shape=(1+2*n_horizons+1,) order=
  [intercept, rv_betas(n_horizons), rs_betas(n_horizons), jump_beta]
- Equivalence to extended OLS (positive rv/rs so ReLU inactive): max_diff ≈ 2e-7 < 1e-4 tol
- `HARNetExtendedTrainer.predict(rv_full, rs_full, dj_full, n_train)` takes *normalised* series;
  returns test predictions in scale_rv space (unscale with `unscale_rv` before metric computation)

### generate_extended_data (`data/generate_extended_data.py`)
- Calls `load_oxford_man_data()` for RV; draws RS- noise with `np.random.default_rng(2024)`
- Single shared RNG iterates over tickers in order (SPX→FTSE→DJI); changing order changes results
- RS- = clip(0.40*rv + N(0,(0.05*rv)²), 0.01*rv, 0.99*rv); ΔJ = rv - 2*rs_neg
- Returns `{ticker: pd.DataFrame}` with columns `['rv', 'rs_neg', 'delta_j']`, DatetimeIndex

### Exp5 Extended Feature Convention
- Extended OLS regressors: [scale_rv(rv)_h, scale_rv(rs_neg)_h, delta_j[t-1]/RV_SCALE]
  for h in horizons; built with `build_har_features` from `src/transforms.py`
- dj_feat = `delta_j[burn_in-1:n-1] / RV_SCALE` — aligns with HARNet's causal lag-1 slice
- Verify equivalence before training: max_diff < 1e-4 asserted in `verify_equivalence()`

