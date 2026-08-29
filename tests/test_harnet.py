"""Unit tests for src/harnet.py (HARNet and HARNetTrainer)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.har_model import HARBaseline
from src.harnet import HARNet, HARNetTrainer
from src.losses import qlike_loss_torch
from src.transforms import scale_rv

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N = 120
_RV = np.exp(_RNG.normal(-7.0, 0.3, size=_N))   # realistic positive RV
_Z = scale_rv(_RV)                                 # scaled series


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------

def test_har20_parameter_count():
    """HAR20 must have exactly 13 trainable parameters."""
    net = HARNet(horizons=(1, 5, 20))
    assert net.parameter_count == 13, (
        f"Expected 13 parameters for HAR20, got {net.parameter_count}"
    )


def test_har80_parameter_count():
    """HAR80 must have exactly 19 trainable parameters."""
    net = HARNet(horizons=(1, 5, 20, 40, 80))
    assert net.parameter_count == 19, (
        f"Expected 19 parameters for HAR80, got {net.parameter_count}"
    )


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------

def test_forward_output_shape_har20():
    """Forward pass should return (batch, T - burn_in) for HAR20."""
    net = HARNet(horizons=(1, 5, 20))
    burn_in = 20
    batch, T = 3, 50
    z = torch.randn(batch, T)
    out = net(z)
    expected = (batch, T - burn_in)
    assert out.shape == expected, f"Expected shape {expected}, got {out.shape}"


def test_forward_output_shape_har80():
    """Forward pass should return (batch, T - burn_in) for HAR80."""
    net = HARNet(horizons=(1, 5, 20, 40, 80))
    burn_in = 80
    batch, T = 2, 100
    z = torch.randn(batch, T)
    out = net(z)
    expected = (batch, T - burn_in)
    assert out.shape == expected, f"Expected shape {expected}, got {out.shape}"


# ---------------------------------------------------------------------------
# Baseline equivalence
# ---------------------------------------------------------------------------

def _check_equivalence(horizons: tuple, tol: float = 1e-5) -> float:
    """Fit OLS baseline, init HARNet, compare predictions. Return max diff."""
    baseline = HARBaseline("OLS", horizons=horizons)
    baseline.fit(_RV)

    # Baseline predictions in scaled-z space
    z_pred_baseline = baseline.predict(_RV)  # shape (N - burn_in,)

    net = HARNet(horizons=horizons)
    net.init_from_baseline(baseline.coef_)

    z_t = torch.tensor(_Z, dtype=torch.float32).unsqueeze(0)  # (1, N)
    net.eval()
    with torch.no_grad():
        z_pred_net = net(z_t).squeeze(0).numpy()  # (N - burn_in,)

    max_diff = float(np.max(np.abs(z_pred_baseline - z_pred_net)))
    assert max_diff < tol, (
        f"Baseline equivalence check failed for horizons={horizons}: "
        f"max_diff={max_diff:.2e} > tol={tol:.2e}"
    )
    return max_diff


def test_baseline_equivalence_har20():
    """After OLS init, HAR20 must replicate baseline predictions to within 1e-5."""
    _check_equivalence((1, 5, 20))


def test_baseline_equivalence_har80():
    """After OLS init, HAR80 must replicate baseline predictions to within 1e-5."""
    _check_equivalence((1, 5, 20, 40, 80))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _full_qlike(net: HARNet, z: np.ndarray) -> float:
    """QLIKE of the full series against its own one-step predictions."""
    net.eval()
    burn_in = net.horizons[-1]
    z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        preds = net(z_t)
        targets = z_t[:, burn_in:]
        loss = qlike_loss_torch(targets, preds)
    return float(loss.item())


def test_training_reduces_loss():
    """After 100 QLIKE iterations from OLS init the loss should not increase."""
    net = HARNet(horizons=(1, 5, 20))
    baseline = HARBaseline("OLS", horizons=(1, 5, 20))
    baseline.fit(_RV)
    net.init_from_baseline(baseline.coef_)

    init_loss = _full_qlike(net, _Z)

    trainer = HARNetTrainer(
        net,
        loss_name="QLIKE",
        lr=1e-4,
        batch_size=4,
        seg_len=5,
        n_iter=100,
        seed=0,
    )
    trainer.train(_Z)
    final_loss = _full_qlike(net, _Z)

    assert final_loss <= init_loss + 1e-4, (
        f"Training increased QLIKE: init={init_loss:.4f}, final={final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# HARNetTrainer.predict
# ---------------------------------------------------------------------------

def test_trainer_predict_shape():
    """trainer.predict should return an array of length len(z) - start_idx."""
    net = HARNet(horizons=(1, 5, 20))
    baseline = HARBaseline("OLS", horizons=(1, 5, 20))
    baseline.fit(_RV)
    net.init_from_baseline(baseline.coef_)

    trainer = HARNetTrainer(net, loss_name="QLIKE", n_iter=5, seed=0)
    trainer.train(_Z)

    n_train = 80
    preds = trainer.predict(_Z, start_idx=n_train)
    expected_len = len(_Z) - n_train
    assert len(preds) == expected_len, (
        f"Expected predict length {expected_len}, got {len(preds)}"
    )


# ---------------------------------------------------------------------------
# get_effective_daily_weights
# ---------------------------------------------------------------------------

def test_effective_weights_har20_length():
    """get_effective_daily_weights should return n_horizons (= 3) tuples for HAR20."""
    net = HARNet(horizons=(1, 5, 20))
    eff = net.get_effective_daily_weights()
    assert len(eff) == 3, f"Expected 3 feature tuples, got {len(eff)}"


def test_effective_weights_har20_daily():
    """Daily feature (index 0) should be lag [1] with weight [1.0]."""
    net = HARNet(horizons=(1, 5, 20))
    lags, weights = net.get_effective_daily_weights()[0]
    assert list(lags) == [1], f"Daily lags should be [1], got {list(lags)}"
    np.testing.assert_allclose(weights, [1.0], atol=1e-12)


def test_effective_weights_har20_weekly_lags():
    """Weekly feature (index 1) should cover lags 1-5."""
    net = HARNet(horizons=(1, 5, 20))
    lags, _ = net.get_effective_daily_weights()[1]
    assert set(lags) == set(range(1, 6)), (
        f"Weekly feature should cover lags 1-5, got {sorted(lags)}"
    )


def test_effective_weights_har20_monthly_lags():
    """Monthly feature (index 2) should cover lags 1-20."""
    net = HARNet(horizons=(1, 5, 20))
    lags, _ = net.get_effective_daily_weights()[2]
    assert set(lags) == set(range(1, 21)), (
        f"Monthly feature should cover lags 1-20, got {sorted(lags)}"
    )


def test_effective_weights_uniform_sum_to_one():
    """With uniform (default) filters, each feature's weights should sum to 1."""
    net = HARNet(horizons=(1, 5, 20))
    eff = net.get_effective_daily_weights()
    for i, (_, weights) in enumerate(eff):
        total = float(np.sum(weights))
        assert abs(total - 1.0) < 1e-6, (
            f"Feature {i} weights should sum to 1.0 with uniform init, got {total:.6f}"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_horizons_validation_must_start_with_1():
    """HARNet with horizons not starting at 1 should raise ValueError."""
    with pytest.raises(ValueError, match="must start with 1"):
        HARNet(horizons=(2, 5, 20))


def test_horizons_validation_must_be_increasing():
    """HARNet with non-strictly-increasing horizons should raise ValueError."""
    with pytest.raises(ValueError, match="strictly increasing"):
        HARNet(horizons=(1, 5, 5))
