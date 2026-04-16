"""LSTM sequence model for SPY next-day prediction.

XGBoost on engineered features is strong, but it only sees each day as an
independent row. An LSTM can learn *temporal patterns* directly from the
raw sequence of cross-asset returns — e.g. "three days of rising VIX + falling
HYG + flat SPY historically precedes a drawdown."

The model has two output heads so one forward pass gives us both a regression
estimate (next-day log return) and a classification probability (up/down).

Implementation notes:
  * Trains on CPU by default (small model, ~3-5s/epoch).
  * Standardizes features using ONLY the training split to prevent leakage.
  * Uses chronological train/test split (same as XGBoost path).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:  # Torch is optional — fall back gracefully if not installed
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

from .config import MODEL_DIR, TARGET_TICKER, ModelConfig

logger = logging.getLogger(__name__)

SEQ_LEN = 30  # trading days of history the LSTM sees per prediction


@dataclass
class LSTMResult:
    """Metrics from an LSTM training run."""

    reg_rmse: float
    reg_direction_acc: float
    clf_accuracy: float
    clf_auc: float
    n_train: int
    n_test: int
    epochs: int
    final_loss: float

    def to_dict(self) -> dict:
        return {k: float(v) if isinstance(v, (int, float)) else v for k, v in self.__dict__.items()}


def _daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def _build_sequences(
    returns: pd.DataFrame,
    target_returns: pd.Series,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Turn a returns panel into (X_seq, y_reg, y_clf) training arrays.

    X_seq[i] = returns.iloc[i-seq_len+1 : i+1] (shape seq_len x n_features)
    y[i]     = target_returns.iloc[i+1]      (next-day target)
    """
    data = returns.dropna().values
    targets = target_returns.reindex(returns.index).values
    index = returns.index

    X_seq, y_reg, y_clf, idx = [], [], [], []
    for t in range(seq_len - 1, len(data) - 1):
        window = data[t - seq_len + 1 : t + 1]
        tgt = targets[t + 1]
        if np.isnan(window).any() or np.isnan(tgt):
            continue
        X_seq.append(window)
        y_reg.append(tgt)
        y_clf.append(1.0 if tgt > 0 else 0.0)
        idx.append(index[t + 1])
    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_reg, dtype=np.float32),
        np.asarray(y_clf, dtype=np.float32),
        pd.DatetimeIndex(idx),
    )


if TORCH_AVAILABLE:

    class SPYLSTM(nn.Module):
        """Two-layer LSTM with two linear heads (regression + classification)."""

        def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
            )
            self.reg_head = nn.Linear(hidden, 1)
            self.clf_head = nn.Linear(hidden, 1)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            out, _ = self.lstm(x)
            last = out[:, -1, :]  # final timestep's hidden state
            return self.reg_head(last).squeeze(-1), self.clf_head(last).squeeze(-1)


@dataclass
class LSTMArtifact:
    """Everything we need to reload a trained LSTM and run inference."""

    model_state: dict
    feature_columns: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    seq_len: int
    hidden: int
    metrics: dict


def train_lstm(
    prices: pd.DataFrame,
    cfg: Optional[ModelConfig] = None,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 64,
    seq_len: int = SEQ_LEN,
) -> tuple[Optional["SPYLSTM"], LSTMArtifact, LSTMResult]:
    """Train the LSTM and return model + serializable artifact + metrics."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. `pip install torch` to enable the LSTM."
        )
    cfg = cfg or ModelConfig()

    returns = _daily_log_returns(prices).dropna()
    target = returns[TARGET_TICKER]

    X, yr, yc, idx = _build_sequences(returns, target, seq_len)
    if len(X) < 200:
        raise ValueError(f"Not enough sequences to train LSTM: got {len(X)}")

    # Chronological split — never shuffle time series
    split = int(len(X) * (1 - cfg.test_size))
    Xtr, Xte = X[:split], X[split:]
    yr_tr, yr_te = yr[:split], yr[split:]
    yc_tr, yc_te = yc[:split], yc[split:]

    # Standardize using train stats only
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-8
    Xtr_n = (Xtr - mean) / std
    Xte_n = (Xte - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SPYLSTM(n_features=X.shape[-1], hidden=hidden).to(device)

    train_ds = TensorDataset(
        torch.tensor(Xtr_n), torch.tensor(yr_tr), torch.tensor(yc_tr)
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    last_loss = float("nan")
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yrb, ycb in loader:
            xb, yrb, ycb = xb.to(device), yrb.to(device), ycb.to(device)
            opt.zero_grad()
            p_reg, p_clf_logits = model(xb)
            loss = mse(p_reg, yrb) + bce(p_clf_logits, ycb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        last_loss = total / len(train_ds)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info("LSTM epoch %2d/%d  loss=%.5f", epoch + 1, epochs, last_loss)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(Xte_n).to(device)
        p_reg, p_clf_logits = model(xt)
        p_reg = p_reg.cpu().numpy()
        p_prob = torch.sigmoid(p_clf_logits).cpu().numpy()

    rmse = float(np.sqrt(np.mean((p_reg - yr_te) ** 2)))
    dir_acc = float(np.mean(np.sign(p_reg) == np.sign(yr_te)))
    clf_pred = (p_prob > 0.5).astype(int)
    clf_acc = float(np.mean(clf_pred == yc_te))
    # ROC-AUC without importing sklearn here
    auc = _roc_auc(yc_te, p_prob)

    result = LSTMResult(
        reg_rmse=rmse,
        reg_direction_acc=dir_acc,
        clf_accuracy=clf_acc,
        clf_auc=auc,
        n_train=len(Xtr),
        n_test=len(Xte),
        epochs=epochs,
        final_loss=last_loss,
    )

    artifact = LSTMArtifact(
        model_state={k: v.cpu() for k, v in model.state_dict().items()},
        feature_columns=list(returns.columns),
        feature_mean=mean,
        feature_std=std,
        seq_len=seq_len,
        hidden=hidden,
        metrics=result.to_dict(),
    )
    return model, artifact, result


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Small in-house ROC-AUC so this module doesn't depend on sklearn."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    pos = y_sorted.sum()
    neg = len(y_sorted) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / pos
    fpr = fp / neg
    return float(np.trapz(tpr, fpr))


def save_lstm(artifact: LSTMArtifact, path: Path | None = None) -> Path:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed")
    path = path or (MODEL_DIR / "lstm.pt")
    torch.save(
        {
            "model_state": artifact.model_state,
            "feature_columns": artifact.feature_columns,
            "feature_mean": artifact.feature_mean,
            "feature_std": artifact.feature_std,
            "seq_len": artifact.seq_len,
            "hidden": artifact.hidden,
            "metrics": artifact.metrics,
        },
        path,
    )
    (MODEL_DIR / "lstm_metrics.json").write_text(json.dumps(artifact.metrics, indent=2))
    return path


def load_lstm(path: Path | None = None) -> tuple[Optional["SPYLSTM"], Optional[LSTMArtifact]]:
    if not TORCH_AVAILABLE:
        return None, None
    path = path or (MODEL_DIR / "lstm.pt")
    if not path.exists():
        return None, None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = SPYLSTM(n_features=len(blob["feature_columns"]), hidden=blob["hidden"])
    model.load_state_dict(blob["model_state"])
    model.eval()
    artifact = LSTMArtifact(
        model_state=blob["model_state"],
        feature_columns=blob["feature_columns"],
        feature_mean=np.asarray(blob["feature_mean"]),
        feature_std=np.asarray(blob["feature_std"]),
        seq_len=blob["seq_len"],
        hidden=blob["hidden"],
        metrics=blob["metrics"],
    )
    return model, artifact


def predict_lstm_next(
    model: "SPYLSTM",
    artifact: LSTMArtifact,
    prices: pd.DataFrame,
) -> dict:
    """Run the LSTM on the most recent sequence and return a prediction dict."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed")
    returns = _daily_log_returns(prices)[artifact.feature_columns].dropna()
    if len(returns) < artifact.seq_len:
        raise ValueError("Not enough recent history to form an LSTM input sequence")

    window = returns.tail(artifact.seq_len).values.astype(np.float32)
    window_n = (window - artifact.feature_mean) / artifact.feature_std
    xb = torch.tensor(window_n).unsqueeze(0)  # (1, seq_len, n_features)
    model.eval()
    with torch.no_grad():
        p_reg, p_logit = model(xb)
        ret = float(p_reg.item())
        prob = float(torch.sigmoid(p_logit).item())

    return {
        "expected_log_return": ret,
        "expected_pct_return": float(np.expm1(ret)) * 100,
        "prob_up": prob,
        "direction": "UP" if prob >= 0.5 else "DOWN",
        "confidence": abs(prob - 0.5) * 2,
        "as_of": returns.index[-1].strftime("%Y-%m-%d"),
    }
