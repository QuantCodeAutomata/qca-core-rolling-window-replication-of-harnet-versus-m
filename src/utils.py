"""General-purpose utilities for the HARNet project."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """
    Set random seeds for NumPy, PyTorch, and Python's ``random`` module.

    Parameters
    ----------
    seed : int
        Seed value used across all RNG backends.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def to_numpy(x) -> np.ndarray:
    """
    Convert *x* to a NumPy array.

    If *x* is a ``torch.Tensor`` it is detached from the computation graph and
    moved to CPU before conversion.  Any other array-like object is passed
    directly to ``np.array``.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)
