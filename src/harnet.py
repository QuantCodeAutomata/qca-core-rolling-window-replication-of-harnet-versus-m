"""
HARNet: Heterogeneous AutoRegressive Network for realized volatility forecasting.

Architecture
------------
HARNet generalises the linear HAR model by replacing the fixed averaging
kernels with learnable causal dilated filters stacked in a pyramid.

For horizons = (1, 5, 20) — "HAR20":

    z           daily series (scaled)
    x1[t]     = z[t-1]                                    (identity, no params)
    h2[t]     = Σ_j w2[j] · z[t-(j+1)]   j=0..4         (5 params, dilation 1)
    h3[t]     = Σ_j w3[j] · relu(h2[t-(j+1)·5])  j=0..3 (4 params, dilation 5)
    ŷ[t]      = β₀ + β₁·x1[t] + β₂·relu(h2[t]) + β₃·relu(h3[t])

    Parameters: 5 + 4 + 4 = 13

For horizons = (1, 5, 20, 40, 80) — "HAR80":

    h4[t] = Σ_j w4[j] · relu(h3[t-(j+1)·20]) j=0,1     (2 params, dilation 20)
    h5[t] = Σ_j w5[j] · relu(h4[t-(j+1)·40]) j=0,1     (2 params, dilation 40)
    ŷ[t] = β₀ + Σ_{k=1}^{5} β_k · f_k[t]

    Parameters: 5 + 4 + 2 + 2 + 6 = 19

The filter lengths are derived from consecutive horizon ratios:
    filter_lengths[k] = horizons[k+1] // horizons[k]

All convolutional layers use causal zero-padding so that the feature at
position t never leaks information from z[t] or later.  A ReLU activation
is applied after each hidden (non-identity) layer.

Vectorised forward pass
-----------------------
For an input z of shape (batch, T), all features are computed for every time
step in parallel using ``F.pad`` + ``Tensor.unfold`` (equivalent to strided
dilated causal correlation), without any Python loop over time steps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import LOSS_FN_TORCH


class HARNet(nn.Module):
    """
    Learnable dilated causal filter network for HAR realized-variance forecasting.

    Parameters
    ----------
    horizons : tuple of int
        Sequence of multiplicative horizons, e.g. ``(1, 5, 20)`` (HAR20) or
        ``(1, 5, 20, 40, 80)`` (HAR80).  Must start with 1 and be strictly
        increasing.  Consecutive ratios define the filter lengths.
    """

    def __init__(self, horizons: tuple = (1, 5, 20)) -> None:
        super().__init__()

        if horizons[0] != 1:
            raise ValueError("horizons must start with 1 (daily).")
        if any(horizons[k + 1] <= horizons[k] for k in range(len(horizons) - 1)):
            raise ValueError("horizons must be strictly increasing.")

        self.horizons = horizons
        self.n_horizons = len(horizons)

        # Filter lengths from consecutive ratios: [h1/h0, h2/h1, ...]
        self.filter_lengths: list[int] = [
            horizons[k + 1] // horizons[k] for k in range(len(horizons) - 1)
        ]

        # One learnable parameter vector per hidden layer (layer 2 onward)
        self.filters = nn.ParameterList(
            [
                nn.Parameter(
                    torch.full((fl,), 1.0 / fl),  # initialised to uniform average
                    requires_grad=True,
                )
                for fl in self.filter_lengths
            ]
        )

        # Output linear layer: (n_horizons features) -> scalar
        # Bias = β₀, weight vector = [β₁, …, β_L]
        self.out_layer = nn.Linear(self.n_horizons, 1, bias=True)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        One-step-ahead predictions for all valid positions in *z*.

        Causal tap convention
        ---------------------
        Layer 1 (on z): taps at z[t-1], z[t-2], …, z[t-fl]  (delay 1..fl)
        Layer k≥2 (on h_{k-1}): taps at h[t], h[t-dil], …, h[t-(fl-1)*dil] (delay 0..(fl-1)*dil)

        The asymmetry preserves baseline equivalence:
            h2[t]  = mean(z[t-1:t-5])   → weekly average
            h3[t]  = mean(h2[t], h2[t-5], h2[t-10], h2[t-15])
                   = (1/20)*sum_{n=1}^{20} z[t-n]   → monthly average ✓

        Parameters
        ----------
        z : torch.Tensor, shape (batch, T)
            Scaled daily RV series.  Must have T > max(horizons).

        Returns
        -------
        torch.Tensor, shape (batch, T - burn_in)
            Predictions for positions burn_in … T-1 on the scaled RV space.
            Prediction at output index i corresponds to forecasting z[:, burn_in+i].
        """
        batch, T = z.shape
        burn_in = self.horizons[-1]

        # dilations[k] = horizons[k] = step size between taps for layer k+1
        dilations = list(self.horizons[:-1])  # [1, 5, 20, …]

        # Build hidden feature sequences (each shape: (batch, T))
        current = z
        hidden_seqs: list[torch.Tensor] = []

        for k, (fl, dil, w) in enumerate(zip(self.filter_lengths, dilations, self.filters)):
            if k == 0:
                # First layer: taps at delay 1, 2, …, fl*dil (shift by one extra dil)
                pad = fl * dil
            else:
                # Subsequent layers: taps at delay 0, dil, …, (fl-1)*dil (no extra shift)
                pad = (fl - 1) * dil

            current_pad = F.pad(current, (pad, 0))  # (batch, T + pad)

            # Gather taps using the shared index formula:
            # slice (fl-1-j)*dil : (fl-1-j)*dil + T  →  delay j*dil + (shift difference)
            taps = torch.stack(
                [
                    current_pad[:, (fl - 1 - j) * dil : (fl - 1 - j) * dil + T]
                    for j in range(fl)
                ],
                dim=2,
            )  # (batch, T, fl)

            new_h = (taps * w).sum(dim=2)  # (batch, T)
            new_h = F.relu(new_h)
            current = new_h
            hidden_seqs.append(new_h)

        # ---------------------------------------------------------------
        # Assemble features for valid positions t = burn_in … T-1
        # ---------------------------------------------------------------
        n_valid = T - burn_in

        # Daily identity feature: x1[t] = z[t-1]
        x1 = z[:, burn_in - 1 : T - 1]  # (batch, n_valid)

        # Hidden features at valid positions
        h_valid = [h[:, burn_in:T] for h in hidden_seqs]  # each (batch, n_valid)

        features = torch.stack([x1] + h_valid, dim=2)  # (batch, n_valid, n_horizons)

        # Linear output layer
        out = self.out_layer(features)  # (batch, n_valid, 1)
        return out.squeeze(-1)  # (batch, n_valid)

    # ------------------------------------------------------------------
    # Initialisation from a baseline HAR solution
    # ------------------------------------------------------------------

    def init_from_baseline(self, coef: np.ndarray) -> None:
        """
        Warm-start from HAR-OLS (or similar) coefficient estimates.

        Filter weights are reset to the uniform average 1/filter_length.
        The output layer is set so that the network reproduces the linear HAR
        prediction when all filter activations are in the linear regime.

        Parameters
        ----------
        coef : np.ndarray, shape (1 + n_horizons,)
            Coefficients [intercept, β₁, …, β_L] from a fitted
            :class:`~src.har_model.HARBaseline`.
        """
        with torch.no_grad():
            # Reset filters to uniform averages
            for fl, w in zip(self.filter_lengths, self.filters):
                w.data.fill_(1.0 / fl)

            # Output layer
            self.out_layer.bias.data.fill_(float(coef[0]))
            self.out_layer.weight.data = torch.tensor(
                coef[1:], dtype=torch.float32
            ).unsqueeze(0)  # (1, n_horizons)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def get_effective_daily_weights(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Compute the effective weights each feature places on individual daily lags.

        The ReLU non-linearities are treated as linear (valid near their
        operating region) to allow a clean linear decomposition.

        Returns
        -------
        list of (lags, weights) tuples, one per horizon feature.
        Each ``lags`` array contains the integer daily lags (1 = yesterday, …)
        and ``weights`` the corresponding filter coefficients.

        Index 0 → daily identity feature (lag [1], weight [1.0])
        Index 1 → weekly feature (lags [1…fl2], weights w2)
        Index 2 → monthly feature (linear composition of w3 ∘ w2)
        …
        """
        w_list = [p.detach().cpu().numpy() for p in self.filters]
        dilations = list(self.horizons[:-1])

        result: list[tuple[np.ndarray, np.ndarray]] = []

        # Feature 0: daily identity
        result.append((np.array([1]), np.array([1.0])))

        # Feature 1: first filter layer directly on z
        fl0, dil0 = self.filter_lengths[0], dilations[0]
        lags0 = np.array([(j + 1) * dil0 for j in range(fl0)])
        weights0 = w_list[0].copy()
        result.append((lags0, weights0))

        # Features 2…: successive compositions
        # Layer k>=1 uses taps at delays 0, dil, 2*dil, ... (j*dil) not (j+1)*dil
        curr_lags = lags0.copy()
        curr_weights = weights0.copy()

        for k in range(1, len(self.filter_lengths)):
            fl_k = self.filter_lengths[k]
            dil_k = dilations[k]
            w_k = w_list[k]

            new_lags_list = []
            new_weights_list = []
            for j in range(fl_k):
                extra = j * dil_k  # delay 0, dil, 2*dil, … for layers k>=1
                for lag_prev, w_prev in zip(curr_lags, curr_weights):
                    new_lags_list.append(lag_prev + extra)
                    new_weights_list.append(float(w_k[j]) * float(w_prev))

            curr_lags = np.array(new_lags_list)
            curr_weights = np.array(new_weights_list)
            result.append((curr_lags, curr_weights))

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"HARNet(horizons={self.horizons}, "
            f"filter_lengths={self.filter_lengths}, "
            f"parameters={self.parameter_count})"
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class HARNetTrainer:
    """
    Mini-batch stochastic trainer for :class:`HARNet`.

    Each training iteration randomly samples a contiguous segment from the
    training series, runs the forward pass, and back-propagates through the
    specified loss function using the Adam optimiser.

    Parameters
    ----------
    model : HARNet
        The model to train (modified in-place).
    loss_name : str
        One of ``'MAE'``, ``'MSE'``, ``'QLIKE'``.
    lr : float
        Adam learning rate.
    batch_size : int
        Number of segments sampled per iteration.
    seg_len : int
        Number of *target* time steps per segment (the segment fed to the
        model has length ``seg_len + burn_in``).
    n_iter : int
        Number of gradient update steps.
    seed : int
        Master seed for PyTorch and NumPy RNGs.
    """

    def __init__(
        self,
        model: HARNet,
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

    def train(self, z_train: np.ndarray) -> list[float]:
        """
        Train the model on *z_train* for ``n_iter`` iterations.

        Parameters
        ----------
        z_train : np.ndarray
            Transformed (scaled) daily RV training series.

        Returns
        -------
        list of float
            Training loss at each iteration.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        burn_in = self.model.horizons[-1]
        full_len = self.seg_len + burn_in
        n_train = len(z_train)

        if n_train < full_len:
            raise ValueError(
                f"Training series too short: need ≥ {full_len} points, got {n_train}."
            )

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        losses: list[float] = []

        self.model.train()
        for _ in range(self.n_iter):
            # Sample batch_size random starting positions
            max_start = n_train - full_len
            starts = np.random.randint(0, max_start + 1, size=self.batch_size)

            segments = np.stack(
                [z_train[s : s + full_len] for s in starts], axis=0
            )  # (batch, full_len)

            seg_tensor = torch.tensor(segments, dtype=torch.float32)

            # Forward pass → predictions for seg_len valid positions
            preds = self.model(seg_tensor)  # (batch, seg_len)
            targets = seg_tensor[:, burn_in:]  # (batch, seg_len)

            loss = self.loss_fn(targets, preds)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            losses.append(float(loss.item()))

        self.model.eval()
        return losses

    def predict(self, z_series: np.ndarray, start_idx: int) -> np.ndarray:
        """
        One-step-ahead predictions for positions ``start_idx`` … ``T-1``.

        The full *z_series* is fed to the model so that every prediction has
        access to the maximum available context.

        Parameters
        ----------
        z_series : np.ndarray
            Transformed (scaled) daily RV series (can be longer than training).
        start_idx : int
            First position to return predictions for.  Must be ≥
            ``max(model.horizons)`` (the burn-in).

        Returns
        -------
        np.ndarray
            Predictions on the transformed scale, length
            ``len(z_series) - start_idx``.
        """
        burn_in = self.model.horizons[-1]
        if start_idx < burn_in:
            raise ValueError(
                f"start_idx ({start_idx}) must be ≥ burn_in ({burn_in})."
            )

        z_tensor = torch.tensor(z_series, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(z_tensor)  # (1, T - burn_in)

        preds_np = preds.squeeze(0).cpu().numpy()

        # preds_np[i] predicts z_series[burn_in + i]
        offset = start_idx - burn_in
        return preds_np[offset:]
