"""Unit tests for src/losses.py — NumPy and PyTorch loss functions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.losses import (
    LOSS_FN_NP,
    LOSS_FN_TORCH,
    mae_loss_np,
    mae_loss_torch,
    mse_loss_np,
    mse_loss_torch,
    qlike_loss_np,
    qlike_loss_torch,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)
_Y = np.abs(_RNG.normal(1.0, 0.3, size=50)) + 0.1   # strictly positive
_Y_TORCH = torch.tensor(_Y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# NumPy — zero-loss when prediction equals target
# ---------------------------------------------------------------------------

def test_mae_zero():
    """mae_loss_np(x, x) should be exactly 0.0."""
    result = mae_loss_np(_Y, _Y)
    assert result == 0.0, f"Expected 0.0, got {result}"


def test_mse_zero():
    """mse_loss_np(x, x) should be exactly 0.0."""
    result = mse_loss_np(_Y, _Y)
    assert result == 0.0, f"Expected 0.0, got {result}"


def test_qlike_zero_when_perfect():
    """qlike_loss_np(x, x) should be approximately 0.0 (ratio 1, log 0)."""
    result = qlike_loss_np(_Y, _Y)
    assert abs(result) < 1e-6, f"Expected ~0.0 for perfect predictions, got {result}"


# ---------------------------------------------------------------------------
# NumPy — QLIKE properties
# ---------------------------------------------------------------------------

def test_qlike_positive():
    """QLIKE should be non-negative for any strictly positive inputs."""
    y_pred = np.abs(_RNG.normal(1.0, 0.5, size=50)) + 0.01
    result = qlike_loss_np(_Y, y_pred)
    assert result >= 0.0, f"QLIKE should be non-negative, got {result}"


def test_qlike_asymmetry_under_vs_over():
    """
    QLIKE should penalise under-prediction more than over-prediction of the
    same relative magnitude.

    Under: y_pred = 0.5 * y_true  →  ratio=2,  QLIKE=2-ln2-1≈0.307
    Over:  y_pred = 2.0 * y_true  →  ratio=0.5, QLIKE=0.5+ln2-1≈0.193
    """
    y_true = np.ones(20)
    under_pred = 0.5 * y_true
    over_pred = 2.0 * y_true

    qlike_under = qlike_loss_np(y_true, under_pred)
    qlike_over = qlike_loss_np(y_true, over_pred)
    assert qlike_under > qlike_over, (
        f"Under-prediction QLIKE ({qlike_under:.4f}) should exceed "
        f"over-prediction QLIKE ({qlike_over:.4f})"
    )


def test_qlike_known_values():
    """Verify QLIKE against analytically computed values."""
    y_true = np.array([1.0])
    # Under: ratio=2, QLIKE = 2 - log(2) - 1
    y_under = np.array([0.5])
    expected_under = 2.0 - np.log(2.0) - 1.0
    result_under = qlike_loss_np(y_true, y_under)
    assert abs(result_under - expected_under) < 1e-10, (
        f"Expected {expected_under:.8f}, got {result_under:.8f}"
    )

    # Over: ratio=0.5, QLIKE = 0.5 - log(0.5) - 1 = 0.5 + log(2) - 1
    y_over = np.array([2.0])
    expected_over = 0.5 + np.log(2.0) - 1.0
    result_over = qlike_loss_np(y_true, y_over)
    assert abs(result_over - expected_over) < 1e-10, (
        f"Expected {expected_over:.8f}, got {result_over:.8f}"
    )


# ---------------------------------------------------------------------------
# NumPy — no NaN or Inf
# ---------------------------------------------------------------------------

def test_all_np_losses_finite():
    """All NumPy loss functions should return a finite scalar."""
    y_pred = np.abs(_RNG.normal(1.0, 0.3, size=50)) + 0.1
    for name, fn in LOSS_FN_NP.items():
        result = fn(_Y, y_pred)
        assert np.isfinite(result), f"Loss '{name}' returned non-finite value: {result}"


# ---------------------------------------------------------------------------
# PyTorch — match NumPy versions
# ---------------------------------------------------------------------------

def _to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def test_torch_mae_matches_numpy():
    """mae_loss_torch should match mae_loss_np to float32 precision."""
    y_pred = np.abs(_RNG.normal(1.0, 0.3, size=50)) + 0.1
    np_val = mae_loss_np(_Y, y_pred)
    torch_val = float(mae_loss_torch(_to_torch(_Y), _to_torch(y_pred)).item())
    assert abs(np_val - torch_val) < 1e-5, (
        f"MAE mismatch: numpy={np_val:.6f}, torch={torch_val:.6f}"
    )


def test_torch_mse_matches_numpy():
    """mse_loss_torch should match mse_loss_np to float32 precision."""
    y_pred = np.abs(_RNG.normal(1.0, 0.3, size=50)) + 0.1
    np_val = mse_loss_np(_Y, y_pred)
    torch_val = float(mse_loss_torch(_to_torch(_Y), _to_torch(y_pred)).item())
    assert abs(np_val - torch_val) < 1e-5, (
        f"MSE mismatch: numpy={np_val:.6f}, torch={torch_val:.6f}"
    )


def test_torch_qlike_matches_numpy():
    """qlike_loss_torch should match qlike_loss_np to float32 precision."""
    y_pred = np.abs(_RNG.normal(1.0, 0.3, size=50)) + 0.1
    np_val = qlike_loss_np(_Y, y_pred)
    torch_val = float(qlike_loss_torch(_to_torch(_Y), _to_torch(y_pred)).item())
    assert abs(np_val - torch_val) < 1e-5, (
        f"QLIKE mismatch: numpy={np_val:.6f}, torch={torch_val:.6f}"
    )


# ---------------------------------------------------------------------------
# PyTorch — no NaN or Inf
# ---------------------------------------------------------------------------

def test_all_torch_losses_finite():
    """All PyTorch loss functions should return a finite scalar tensor."""
    y_pred = np.abs(_RNG.normal(1.0, 0.3, size=50)) + 0.1
    y_t = _to_torch(_Y)
    p_t = _to_torch(y_pred)
    for name, fn in LOSS_FN_TORCH.items():
        result = fn(y_t, p_t)
        assert torch.isfinite(result), (
            f"Loss '{name}' returned non-finite tensor value: {result.item()}"
        )


# ---------------------------------------------------------------------------
# PyTorch — zero-loss when predictions equal targets
# ---------------------------------------------------------------------------

def test_torch_mae_zero():
    """mae_loss_torch(x, x) should be 0.0."""
    result = float(mae_loss_torch(_Y_TORCH, _Y_TORCH).item())
    assert result == 0.0, f"Expected 0.0, got {result}"


def test_torch_mse_zero():
    """mse_loss_torch(x, x) should be 0.0."""
    result = float(mse_loss_torch(_Y_TORCH, _Y_TORCH).item())
    assert result == 0.0, f"Expected 0.0, got {result}"


def test_torch_qlike_zero_when_perfect():
    """qlike_loss_torch(x, x) should be approximately 0.0."""
    result = float(qlike_loss_torch(_Y_TORCH, _Y_TORCH).item())
    assert abs(result) < 1e-6, f"Expected ~0.0, got {result}"
