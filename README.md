# HARNet Realized Volatility Forecasting — Replication Study

A replication of the **HARNet** (Heterogeneous AutoRegressive Network) model for
daily realized volatility forecasting. HARNet replaces the linear lag aggregation
of the classical HAR model with a small convolutional neural network, preserving
the interpretable multi-horizon structure (daily / weekly / monthly) while
allowing non-linear feature extraction. The network is trained end-to-end with a
choice of loss functions (MAE, MSE, QLIKE) and is evaluated against OLS, WLS, and
log-OLS HAR baselines on a rolling-window scheme.

> **Data note:** The Oxford-Man Institute Realized Library is not publicly
> accessible in this environment. All experiments use **synthetic** realized
> variance series that replicate the format and statistical properties of the
> original SPX, FTSE, and DJI series (business days 2001–2020).

---

## Project Structure

```
project/
├── data/                   # Data generation utilities
│   └── generate_data.py    # Synthetic RV series generator (Oxford-Man format)
├── src/                    # Core implementations
│   ├── har_model.py        # OLS / WLS / log-OLS HAR baselines
│   ├── harnet.py           # Custom PyTorch HARNet (HAR20 & HAR80 variants)
│   ├── transforms.py       # RV scaling helpers (level and log-RV)
│   ├── losses.py           # MAE, MSE, and QLIKE loss functions
│   └── metrics.py          # Out-of-sample evaluation metrics
├── exp/                    # Experiment scripts
│   ├── exp1_rolling_window.py   # Exp 1 — rolling-window HARNet vs HAR baselines
│   ├── exp2_qlike_stability.py  # Exp 2 — QLIKE optimisation stability
│   ├── exp3_cross_metric.py     # Exp 3 — cross-metric evaluation
│   ├── exp4_interpretability.py # Exp 4 — learned filter interpretability
│   └── exp5_extended_input.py   # Exp 5 — extended-input HARNet
├── tests/                  # Pytest test suite
├── results/                # Output metrics (CSV) and figures (PNG/PDF)
│   └── RESULTS.md          # Auto-populated results summary
└── requirements.txt        # Python dependencies
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.x are recommended. All other dependencies are listed in
`requirements.txt`.

---

## Usage

Run experiments from the repository root:

```bash
# Experiment 1 — rolling-window HARNet vs HAR baselines
python exp/exp1_rolling_window.py

# Experiment 2 — QLIKE optimisation stability
python exp/exp2_qlike_stability.py

# Experiment 3 — cross-metric evaluation of preferred HARNet (OLS + QLIKE)
python exp/exp3_cross_metric.py

# Experiment 4 — interpretability of learned HARNet filters
python exp/exp4_interpretability.py

# Experiment 5 — extended-input HARNet (RS- and signed jump variation)
python exp/exp5_extended_input.py
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## Experiments

### Experiment 1 — Rolling-Window HARNet vs HAR Baselines
Trains HARNet and three HAR baselines (OLS, WLS, log-OLS) across **15 rolling
windows** (4-year train / 1-year test, 2002–2005/2006 through 2016–2019/2020) on
**three equity indexes** (SPX, FTSE, DJI) under **three loss functions** (MAE,
MSE, QLIKE). Reports out-of-sample MAE, MSE, and QLIKE for every combination.

### Experiment 2 — QLIKE Optimisation Stability
Compares QLIKE-loss training starting from an **HAR-initialised** network versus
**random initialisation**, across multiple random seeds. Examines convergence
stability and sensitivity to initialisation when optimising the asymmetric QLIKE
criterion.

### Experiment 3 — Cross-Metric Evaluation of Preferred HARNet
Takes the best-performing HARNet configuration identified in Exp 1 (OLS
initialisation + QLIKE loss) and evaluates it against all baselines under all
three metrics simultaneously, providing a unified comparison table.

### Experiment 4 — Interpretability of Learned HARNet Filters
Inspects the convolutional filters learned by HARNet and compares their effective
lag weights to the theoretical HAR lag structure (1-day, 1-week, 1-month
averages). Produces visualisations of filter responses alongside classical HAR
coefficients.

### Experiment 5 — Extended-Input HARNet with RS- and Signed Jump Variation
Augments HARNet's input with **realised semi-variance** (downside, RS⁻) and
**signed jump variation** as additional channels, testing whether richer
microstructure features improve forecast accuracy beyond the standard RV input.

---

## References

- Bucci, A. (2020). *Realized Volatility Forecasting with Neural Networks.*
  Journal of Financial Econometrics, 18(3), 502–531.
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized
  Volatility.* Journal of Financial Econometrics, 7(2), 174–196.
- Oxford-Man Institute Realized Library:
  https://realized.oxford-man.ox.ac.uk/
