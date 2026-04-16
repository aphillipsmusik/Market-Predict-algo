"""Ensemble predictions from the XGBoost and LSTM models.

Two very different model families — a tree ensemble on engineered features
and a recurrent net on raw return sequences — tend to make *different*
mistakes. Averaging them is a cheap, reliable way to get a small accuracy
bump and a better-calibrated probability.

Weights default to 50/50 but can be biased (e.g. if the LSTM underperforms
on recent data, down-weight it in the dashboard).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def blend_predictions(
    xgb_pred: dict,
    lstm_pred: Optional[dict] = None,
    w_xgb: float = 0.5,
    w_lstm: float = 0.5,
) -> dict:
    """Blend XGB + LSTM prediction dicts into a single ensemble prediction.

    Falls back to whichever prediction is available if the other is missing.
    """
    if lstm_pred is None:
        return {**xgb_pred, "source": "xgboost"}
    if xgb_pred is None:
        return {**lstm_pred, "source": "lstm"}

    total = w_xgb + w_lstm
    wx, wl = w_xgb / total, w_lstm / total

    ret = wx * xgb_pred["expected_log_return"] + wl * lstm_pred["expected_log_return"]
    prob = wx * xgb_pred["prob_up"] + wl * lstm_pred["prob_up"]

    return {
        "expected_log_return": float(ret),
        "expected_pct_return": float(np.expm1(ret)) * 100,
        "prob_up": float(prob),
        "direction": "UP" if prob >= 0.5 else "DOWN",
        "confidence": float(abs(prob - 0.5) * 2),
        "as_of": xgb_pred.get("as_of") or lstm_pred.get("as_of"),
        "source": "ensemble",
        "components": {"xgb": xgb_pred, "lstm": lstm_pred},
    }
