"""
HARNetExtended: three-channel HARNet for Exp5 (RV + RS- + signed jump variation).

Architecture
------------
Two branches apply the same causal dilated filter pyramid as HARNet:
  - Branch 1 (RV)  : learnable filters rv_filters, producing n_horizons features
  - Branch 2 (RS-) : learnable filters rs_filters, producing n_horizons features
  - Branch 3 (ΔJ)  : Delta J (signed jump) at lag 1, enters the output linearly

Output layer: Linear(2 * n_horizons + 1, 1, bias=True)

The tap convention is identical to HARNet:
  Layer 0 (on z):  taps at delay 1, 2, …, fl     (causal shift by one extra dilation)
  Layer k≥1 (on h): taps at delay 0, dil, …, (fl-1)*dil  (no extra shift)

With uniform filter initialisation this reproduces the linear HAR averages, so
``init_from_extended_baseline`` makes the network exactly replicate an extended
HAR OLS baseline (on the scaled z domain) to within floating-point precision.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import LOSS_FN_TORCH


class HARNetExtended(nn.Module):
    """
    Three-channel HARNet combining RV, RS-, and signed jump variation (ΔJ).

    Parameters
    ----------
    horizons : tuple of int
        Same convention as :class:`~src.harnet.HARNet`, e.g. ``(1, 5, 20)``.
        Must start with 1 and be strictly increasing.
    """

    def __init__(self, horizons: tuple = (1, 5, 20)) -> None:
        super().__init__()

        if horizons[0] != 1:
            raise ValueError("horizons must start with 1 (daily).")
        if any(horizons[k + 1] <= horizons[k] for k in range(len(horizons) - 1)):
            raise ValueError("horizons must be strictly increasing.")

        self.horizons = horizons
        self.n_horizons = len(horizons)

        self.filter_lengths: list[int] = [
            horizons[k + 1] // horizons[k] for k in range(len(horizons) - 1)
        ]

        def _make_filters() -> nn.ParameterList:
            return nn.ParameterList(
                [
                    nn.Parameter(torch.full((fl,), 1.0 / fl), requires_grad=True)
                    for fl in self.filter_lengths
                ]
            )

        self.rv_filters = _make_filters()
        self.rs_filters = _make_filters()

        # Features: n_horizons from RV + n_horizons from RS- + 1 for ΔJ
        self.out_layer = nn.Linear(2 * self.n_horizons + 1, 1, bias=True)

    # ------------------------------------------------------------------
    # Internal filter application
    # ------------------------------------------------------------------

    def _apply_filters(
        self, x: torch.Tensor, filters: nn.ParameterList
    ) -> list[torch.Tensor]:
        """
        Apply the causal dilated filter pyramid to *x*.

        Returns
        -------
        list of torch.Tensor, each shape (batch, T)
            Hidden sequence for each filter layer (after ReLU).
        """
        batch, T = x.shape
        dilations = list(self.horizons[:-1])
        current = x
        hidden_seqs: list[torch.Tensor] = []

        for k, (fl, dil, w) in enumerate(
            zip(self.filter_lengths, dilations, filters)
        ):
            pad = fl * dil if k == 0 else (fl - 1) * dil
            current_pad = F.pad(current, (pad, 0))

            taps = torch.stack(
                [
                    current_pad[:, (fl - 1 - j) * dil : (fl - 1 - j) * dil + T]
                    for j in range(fl)
                ],
                dim=2,
            )  # (batch, T, fl)

            new_h = F.relu((taps * w).sum(dim=2))
            current = new_h
            hidden_seqs.append(new_h)

        return hidden_seqs

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        rv_scaled: torch.Tensor,
        rs_scaled: torch.Tensor,
        dj_lag1: torch.Tensor,
    ) -> torch.Tensor:
        """
        One-step-ahead predictions for all valid positions.

        Parameters
        ----------
        rv_scaled : torch.Tensor, shape (batch, T)
            Scale-rv transformed daily RV series (non-negative).
        rs_scaled : torch.Tensor, shape (batch, T)
            Scale-rv transformed RS- series (non-negative).
        dj_lag1 : torch.Tensor, shape (batch, T)
            Signed jump variation divided by RV_SCALE (can be negative).
            The 1-day lag is applied via the same causal slicing as the daily
            RV feature: ``dj_lag1[:, burn_in-1 : T-1]``.

        Returns
        -------
        torch.Tensor, shape (batch, T - burn_in)
            Predictions in the scaled RV space.
        """
        batch, T = rv_scaled.shape
        burn_in = self.horizons[-1]
        n_valid = T - burn_in

        rv_hidden = self._apply_filters(rv_scaled, self.rv_filters)
        rs_hidden = self._apply_filters(rs_scaled, self.rs_filters)

        # Daily identity features (lag 1 from target)
        rv_x1 = rv_scaled[:, burn_in - 1 : T - 1]  # (batch, n_valid)
        rs_x1 = rs_scaled[:, burn_in - 1 : T - 1]  # (batch, n_valid)

        # Hidden features at valid output positions
        rv_h_valid = [h[:, burn_in:T] for h in rv_hidden]
        rs_h_valid = [h[:, burn_in:T] for h in rs_hidden]

        # ΔJ feature at lag 1 (same causal slice as x1)
        dj_feat = dj_lag1[:, burn_in - 1 : T - 1]  # (batch, n_valid)

        rv_features = torch.stack([rv_x1] + rv_h_valid, dim=2)   # (batch, n_valid, n_horizons)
        rs_features = torch.stack([rs_x1] + rs_h_valid, dim=2)   # (batch, n_valid, n_horizons)
        dj_feature = dj_feat.unsqueeze(-1)                         # (batch, n_valid, 1)

        features = torch.cat([rv_features, rs_features, dj_feature], dim=2)
        # (batch, n_valid, 2*n_horizons+1)

        out = self.out_layer(features)  # (batch, n_valid, 1)
        return out.squeeze(-1)          # (batch, n_valid)

    # ------------------------------------------------------------------
    # Initialisation from extended HAR-OLS baseline
    # ------------------------------------------------------------------

    def init_from_extended_baseline(self, coef: np.ndarray) -> None:
        """
        Warm-start from an extended HAR-OLS coefficient vector.

        Both filter banks are reset to uniform averages so that the network
        reproduces the linear baseline prediction when inputs are non-negative
        (ReLU inactive).

        Parameters
        ----------
        coef : np.ndarray, shape (1 + 2 * n_horizons + 1,)
            Ordered as ``[intercept, rv_betas..., rs_betas..., jump_beta]``.
        """
        with torch.no_grad():
            for fl, w in zip(self.filter_lengths, self.rv_filters):
                w.data.fill_(1.0 / fl)
            for fl, w in zip(self.filter_lengths, self.rs_filters):
                w.data.fill_(1.0 / fl)

            self.out_layer.bias.data.fill_(float(coef[0]))
            self.out_layer.weight.data = torch.tensor(
                coef[1:], dtype=torch.float32
            ).unsqueeze(0)  # (1, 2*n_horizons+1)

    # ------------------------------------------------------------------
    # Properties / repr
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"HARNetExtended(horizons={self.horizons}, "
            f"filter_lengths={self.filter_lengths}, "
            f"parameters={self.parameter_count})"
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class HARNetExtendedTrainer:
    """
    Mini-batch stochastic trainer for :class:`HARNetExtended`.

    Segments are drawn simultaneously from all three input series so that
    temporal alignment is preserved.  The prediction target is the scaled RV
    series (rv_scaled); the loss is computed in that space.

    Parameters
    ----------
    model : HARNetExtended
    loss_name : str
        One of ``'MAE'``, ``'MSE'``, ``'QLIKE'``.
    lr : float
        Adam learning rate.
    batch_size : int
        Segments per iteration.
    seg_len : int
        Number of target time steps per segment.
    n_iter : int
        Gradient update steps.
    seed : int
        Master RNG seed.
    """

    def __init__(
        self,
        model: HARNetExtended,
        loss_name: str = "QLIKE",
        lr: float = 1e-4,
        batch_size: int = 4,
        seg_len: int = 5,
        n_iter: int = 10000,
        seed: int = 42,
    ) -> None:
        if loss_name not in LOSS_FN_TORCH:
            raise ValueError(
                f"loss_name must be one of {list(LOSS_FN_TORCH)}, got '{loss_name}'"
            )
        self.model = model
        self.loss_name = loss_name
        self.loss_fn = LOSS_FN_TORCH[loss_name]
        self.lr = lr
        self.batch_size = batch_size
        self.seg_len = seg_len
        self.n_iter = n_iter
        self.seed = seed

    def train(
        self,
        rv_train: np.ndarray,
        rs_train: np.ndarray,
        dj_train: np.ndarray,
    ) -> list[float]:
        """
        Train on aligned scaled series for ``n_iter`` iterations.

        Parameters
        ----------
        rv_train : np.ndarray
            scale_rv(rv) training series.
        rs_train : np.ndarray
            scale_rv(rs_neg) training series.
        dj_train : np.ndarray
            delta_j / RV_SCALE training series (may be negative).

        Returns
        -------
        list of float
            Training loss at each iteration.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        burn_in = self.model.horizons[-1]
        full_len = self.seg_len + burn_in
        n_train = len(rv_train)

        if n_train < full_len:
            raise ValueError(
                f"Training series too short: need ≥ {full_len} points, got {n_train}."
            )

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        losses: list[float] = []

        self.model.train()
        for _ in range(self.n_iter):
            max_start = n_train - full_len
            starts = np.random.randint(0, max_start + 1, size=self.batch_size)

            rv_segs = np.stack([rv_train[s : s + full_len] for s in starts])
            rs_segs = np.stack([rs_train[s : s + full_len] for s in starts])
            dj_segs = np.stack([dj_train[s : s + full_len] for s in starts])

            rv_t = torch.tensor(rv_segs, dtype=torch.float32)
            rs_t = torch.tensor(rs_segs, dtype=torch.float32)
            dj_t = torch.tensor(dj_segs, dtype=torch.float32)

            preds = self.model(rv_t, rs_t, dj_t)   # (batch, seg_len)
            targets = rv_t[:, burn_in:]              # (batch, seg_len)

            loss = self.loss_fn(targets, preds)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            losses.append(float(loss.item()))

        self.model.eval()
        return losses

    def predict(
        self,
        rv_full: np.ndarray,
        rs_full: np.ndarray,
        dj_full: np.ndarray,
        n_train: int,
    ) -> np.ndarray:
        """
        One-step-ahead predictions for the test period.

        The full concatenated (train + test) series is fed to the model so
        every test prediction benefits from the maximum available context.

        Parameters
        ----------
        rv_full, rs_full, dj_full : np.ndarray
            Full (train + test) scaled series.
        n_train : int
            Length of the training split; test predictions start at this index.

        Returns
        -------
        np.ndarray
            Predictions in scale_rv space, length ``len(rv_full) - n_train``.
        """
        burn_in = self.model.horizons[-1]

        rv_t = torch.tensor(rv_full, dtype=torch.float32).unsqueeze(0)
        rs_t = torch.tensor(rs_full, dtype=torch.float32).unsqueeze(0)
        dj_t = torch.tensor(dj_full, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(rv_t, rs_t, dj_t)  # (1, T - burn_in)

        preds_np = preds.squeeze(0).cpu().numpy()
        offset = n_train - burn_in
        return preds_np[offset:]
