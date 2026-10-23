"""Additional PyTorch models: a GRU over travel-time history, and a quantile network.

Both follow the fairness rules in models.py: preprocessing is fitted on the training fold,
the validation fold alone drives early stopping, and the test fold is never touched.
Both learn the residual (target - last observed) by default, and report scores on the
original scale.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .metrics import mae
from .models import TARGET, Encoder, _base

QUANTILES = (0.1, 0.5, 0.9)


def train_network(net: nn.Module, xs_tr: tuple, y_tr: torch.Tensor, xs_va: tuple, val_score, loss_fn,
                  seed: int, max_epochs: int, patience: int, lr: float, batch_size: int) -> tuple[nn.Module, int]:
    """Shared loop: Adam, mini-batches, early stopping on `val_score(net_output_on_val)` (lower is better)."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, since, epochs_run = float("inf"), copy.deepcopy(net.state_dict()), 0, 0
    for epoch in range(max_epochs):
        net.train()
        order = torch.randperm(len(y_tr), generator=gen)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss_fn(net(*[x[idx] for x in xs_tr]), y_tr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            score = val_score(net(*xs_va))
        epochs_run = epoch + 1
        if score < best:
            best, best_state, since = score, copy.deepcopy(net.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best_state)
    return net, epochs_run


# ---------------------------------------------------------------- GRU over lag history
class GRUNet(nn.Module):
    """GRU reads (value, present-flag) for each lagged bin, oldest first; static features join at the head."""

    def __init__(self, n_static: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)


def _sequence(df: pd.DataFrame, lag_cols: list[str], mean: float, std: float) -> np.ndarray:
    """(n, L, 2) array: scaled value (0 where missing) and a presence flag, oldest bin first."""
    raw = df[lag_cols[::-1]].to_numpy(dtype=float)
    present = ~np.isnan(raw)
    scaled = np.where(present, (raw - mean) / std, 0.0)
    return np.stack([scaled, present.astype(float)], axis=2).astype(np.float32)


@dataclass(frozen=True)
class GRUPredictor:
    encoder: Encoder
    net: nn.Module
    lag_cols: tuple
    seq_mean: float
    seq_std: float
    y_mean: float
    y_std: float
    epochs_run: int
    residual: bool

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        seq = torch.from_numpy(_sequence(df, list(self.lag_cols), self.seq_mean, self.seq_std))
        static = torch.from_numpy(self.encoder.transform(df, impute=True, scale=True))
        self.net.eval()
        with torch.no_grad():
            out = self.net(seq, static).numpy() * self.y_std + self.y_mean
        return out + _base(df, self.residual)


def fit_gru(train: pd.DataFrame, val: pd.DataFrame, static_cols: list[str], lag_cols: list[str], seed: int = 0,
            max_epochs: int = 40, patience: int = 5, lr: float = 2e-3, batch_size: int = 1024,
            residual: bool = True) -> GRUPredictor:
    if val.empty:
        raise ValueError("validation fold is empty")
    if not lag_cols:
        raise ValueError("fit_gru needs lag columns (build_examples(..., n_lags=N))")
    torch.manual_seed(seed)
    lags_all = train[lag_cols].to_numpy(dtype=float)
    seq_mean, seq_std = float(np.nanmean(lags_all)), float(np.nanstd(lags_all) or 1.0)
    enc = Encoder.fit(train, static_cols)
    fit_target = train[TARGET].to_numpy() - _base(train, residual)
    y_mean, y_std = float(fit_target.mean()), float(fit_target.std() or 1.0)
    xs_tr = (torch.from_numpy(_sequence(train, lag_cols, seq_mean, seq_std)),
             torch.from_numpy(enc.transform(train, impute=True, scale=True)))
    xs_va = (torch.from_numpy(_sequence(val, lag_cols, seq_mean, seq_std)),
             torch.from_numpy(enc.transform(val, impute=True, scale=True)))
    y_tr = torch.from_numpy(((fit_target - y_mean) / y_std).astype(np.float32))
    yva, base_va = val[TARGET].to_numpy(), _base(val, residual)

    net, epochs = train_network(
        GRUNet(xs_tr[1].shape[1]), xs_tr, y_tr, xs_va,
        lambda out: mae(yva, out.numpy() * y_std + y_mean + base_va), nn.L1Loss(),
        seed, max_epochs, patience, lr, batch_size)
    return GRUPredictor(enc, net, tuple(lag_cols), seq_mean, seq_std, y_mean, y_std, epochs, residual)


# ---------------------------------------------------------------- quantile network
def pinball_loss(quantiles: tuple):
    q = torch.tensor(quantiles, dtype=torch.float32)

    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = target.unsqueeze(1) - pred
        return torch.maximum(q * err, (q - 1) * err).mean()
    return loss


@dataclass(frozen=True)
class QuantilePredictor:
    encoder: Encoder
    net: nn.Module
    quantiles: tuple
    y_mean: float
    y_std: float
    residual: bool

    def predict_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        """(n, k) predictions, sorted along the quantile axis so intervals never cross."""
        x = torch.from_numpy(self.encoder.transform(df, impute=True, scale=True))
        self.net.eval()
        with torch.no_grad():
            out = self.net(x).numpy() * self.y_std + self.y_mean
        out = out + np.reshape(_base(df, self.residual), (-1, 1))
        return np.sort(out, axis=1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(df)[:, self.quantiles.index(0.5)]


def fit_quantile_mlp(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], seed: int = 0,
                     quantiles: tuple = QUANTILES, max_epochs: int = 40, patience: int = 5, lr: float = 2e-3,
                     batch_size: int = 1024, residual: bool = True) -> QuantilePredictor:
    """MLP with one output per quantile, trained with pinball loss. Early stopping uses the median's MAE."""
    if val.empty:
        raise ValueError("validation fold is empty")
    if 0.5 not in quantiles:
        raise ValueError("quantiles must include 0.5 (the point prediction)")
    torch.manual_seed(seed)
    enc = Encoder.fit(train, cols)
    fit_target = train[TARGET].to_numpy() - _base(train, residual)
    y_mean, y_std = float(fit_target.mean()), float(fit_target.std() or 1.0)
    xs_tr = (torch.from_numpy(enc.transform(train, impute=True, scale=True)),)
    xs_va = (torch.from_numpy(enc.transform(val, impute=True, scale=True)),)
    y_tr = torch.from_numpy(((fit_target - y_mean) / y_std).astype(np.float32))
    yva, base_va, mid = val[TARGET].to_numpy(), _base(val, residual), list(quantiles).index(0.5)
    net = nn.Sequential(nn.Linear(xs_tr[0].shape[1], 64), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(quantiles)))
    net, _ = train_network(net, xs_tr, y_tr, xs_va,
                           lambda out: mae(yva, out[:, mid].numpy() * y_std + y_mean + base_va),
                           pinball_loss(quantiles), seed, max_epochs, patience, lr, batch_size)
    return QuantilePredictor(enc, net, tuple(quantiles), y_mean, y_std, residual)
