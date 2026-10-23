"""Gradient boosting and a small PyTorch network, on identical feature sets.

Fairness rules (research_plan.md section 8):
  - The with-weather and without-weather models differ ONLY in input columns.
  - Every preprocessing statistic (medians, scaling, segment categories) is fitted on
    the training fold alone.
  - Hyperparameters / early stopping are chosen on the validation fold alone. The test
    fold is never seen during fitting.
  - Missing weather is never silently filled for boosting (it handles NaN natively).
    The network gets train-median imputation plus explicit missing-value indicators.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn

from .metrics import mae

TRAFFIC_COLS = ["last_obs_s", "mean4_s", "stale_bins", "tod_min", "dow", "is_weekend"]
WEATHER_COLS = {
    "forecast": ["wx_fc_precipitation", "wx_fc_temperature_2m", "wx_fc_rain"],
    "oracle": ["wx_oracle_precipitation", "wx_oracle_temperature_2m", "wx_oracle_rain"],
}
TARGET = "target_duration_s"
GBM_GRID = [{"max_iter": i, "max_leaf_nodes": n, "learning_rate": 0.05} for i in (100, 300) for n in (8, 31)]


@dataclass(frozen=True)
class Encoder:
    """Turns example rows into a numeric matrix using statistics from the training fold."""
    cols: tuple
    segments: tuple
    medians: tuple
    means: tuple
    stds: tuple
    missing_flag_cols: tuple

    @classmethod
    def fit(cls, train: pd.DataFrame, cols: list[str]) -> "Encoder":
        x = train[cols].astype(float)
        std = x.std(ddof=0).replace(0, 1.0).fillna(1.0)
        return cls(cols=tuple(cols), segments=tuple(sorted(train["segment_id"].unique())),
                   medians=tuple(x.median().fillna(0.0)), means=tuple(x.mean().fillna(0.0)),
                   stds=tuple(std), missing_flag_cols=tuple(c for c in cols if x[c].isna().any()))

    def transform(self, df: pd.DataFrame, impute: bool, scale: bool) -> np.ndarray:
        x = df[list(self.cols)].astype(float)
        flags = [x[c].isna().astype(float).to_numpy() for c in self.missing_flag_cols] if impute else []
        if impute:
            x = x.fillna(dict(zip(self.cols, self.medians)))
        values = x.to_numpy()
        if scale:
            values = (values - np.array(self.means)) / np.array(self.stds)
        onehot = np.column_stack([(df["segment_id"] == s).astype(float).to_numpy() for s in self.segments])
        parts = [values, onehot] + ([np.column_stack(flags)] if flags else [])
        return np.hstack(parts).astype(np.float32)   # unseen segments encode as all zeros


@dataclass(frozen=True)
class GBMPredictor:
    encoder: Encoder
    model: HistGradientBoostingRegressor
    params: dict

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.encoder.transform(df, impute=False, scale=False))


def fit_gbm(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], seed: int = 0) -> GBMPredictor:
    """Gradient boosting with MAE loss; a small grid is searched on the validation fold."""
    if val.empty:
        raise ValueError("validation fold is empty")
    enc = Encoder.fit(train, cols)
    xtr, ytr = enc.transform(train, impute=False, scale=False), train[TARGET].to_numpy()
    xva, yva = enc.transform(val, impute=False, scale=False), val[TARGET].to_numpy()
    best = None
    for params in GBM_GRID:
        model = HistGradientBoostingRegressor(loss="absolute_error", min_samples_leaf=20,
                                              random_state=seed, **params).fit(xtr, ytr)
        score = mae(yva, model.predict(xva))
        if best is None or score < best[0]:
            best = (score, model, params)
    return GBMPredictor(enc, best[1], best[2])


def _make_net(n_inputs: int) -> nn.Module:
    return nn.Sequential(nn.Linear(n_inputs, 64), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


@dataclass(frozen=True)
class MLPPredictor:
    encoder: Encoder
    net: nn.Module
    y_mean: float
    y_std: float
    epochs_run: int

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = torch.from_numpy(self.encoder.transform(df, impute=True, scale=True))
        self.net.eval()
        with torch.no_grad():
            return self.net(x).squeeze(1).numpy() * self.y_std + self.y_mean


def fit_mlp(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], seed: int = 0, max_epochs: int = 200,
            patience: int = 15, lr: float = 1e-3, batch_size: int = 256) -> MLPPredictor:
    """Two-hidden-layer network, L1 loss, early stopping on validation MAE (restores the best epoch)."""
    if val.empty:
        raise ValueError("validation fold is empty")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    enc = Encoder.fit(train, cols)
    y_mean, y_std = float(train[TARGET].mean()), float(train[TARGET].std(ddof=0) or 1.0)
    xtr = torch.from_numpy(enc.transform(train, impute=True, scale=True))
    ytr = torch.from_numpy(((train[TARGET].to_numpy() - y_mean) / y_std).astype(np.float32))
    xva = torch.from_numpy(enc.transform(val, impute=True, scale=True))
    yva = val[TARGET].to_numpy()

    net = _make_net(xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    best_score, best_state, since_best, epochs_run = float("inf"), copy.deepcopy(net.state_dict()), 0, 0
    for epoch in range(max_epochs):
        net.train()
        order = torch.randperm(len(xtr), generator=gen)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss_fn(net(xtr[idx]).squeeze(1), ytr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            score = mae(yva, net(xva).squeeze(1).numpy() * y_std + y_mean)
        epochs_run = epoch + 1
        if score < best_score:
            best_score, best_state, since_best = score, copy.deepcopy(net.state_dict()), 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    net.load_state_dict(best_state)
    return MLPPredictor(enc, net, y_mean, y_std, epochs_run)
