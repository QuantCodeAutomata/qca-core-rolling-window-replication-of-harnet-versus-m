"""
Loss functions for training and evaluation of HAR and HARNet models.

Both NumPy (for offline evaluation) and PyTorch (for gradient-based training)
variants are provided.

QLIKE loss
----------
The QLIKE (Quasi-Likelihood) loss is widely used in volatility forecasting:

    QLIKE(y, y_hat) = mean(y / y_hat - log(y / y_hat) - 1)

It is minimised when y_hat = y and is asymmetric, penalising under-prediction
more heavily than over-prediction.  Both ``y`` and ``y_hat`` must be strictly
positive.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# NumPy versions
# ---------------------------------------------------------------------------

def mae_loss_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mse_loss_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def qlike_loss_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    QLIKE loss: mean(y / y_hat - log(y / y_hat) - 1).

    Requires strictly positive *y_true* and *y_pred*.  A small floor *eps*
    is applied to *y_pred* to prevent division by zero.
    """
    y_pred_safe = np.maximum(y_pred, eps)
    ratio = y_true / y_pred_safe
    return float(np.mean(ratio - np.log(ratio) - 1.0))


# ---------------------------------------------------------------------------
# PyTorch versions
# ---------------------------------------------------------------------------

def mae_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean absolute error (differentiable)."""
    return torch.mean(torch.abs(y_true - y_pred))


def mse_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean squared error (differentiable)."""
    return torch.mean((y_true - y_pred) ** 2)


def qlike_loss_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    QLIKE loss: mean(y / y_hat - log(y / y_hat) - 1).

    A small floor *eps* is applied to *y_pred* to prevent division by zero.
    """
    y_pred_safe = torch.clamp(y_pred, min=eps)
    ratio = y_true / y_pred_safe
    return torch.mean(ratio - torch.log(ratio) - 1.0)


# ---------------------------------------------------------------------------
# Dispatch dictionaries
# ---------------------------------------------------------------------------

LOSS_FN_TORCH = {
    "MAE": mae_loss_torch,
    "MSE": mse_loss_torch,
    "QLIKE": qlike_loss_torch,
}

LOSS_FN_NP = {
    "MAE": mae_loss_np,
    "MSE": mse_loss_np,
    "QLIKE": qlike_loss_np,
}
