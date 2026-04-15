"""Training, evaluation, and inference.

We train two companion models from the same feature set:

  * **Regression** — predicts the *magnitude* of SPY's next-day log return.
  * **Classification** — predicts the *direction* (up/down) as a probability.

Gradient Boosted Trees (XGBoost) is the default because it handles the
mixed-scale, partially-correlated cross-asset features well without heavy
preprocessing. Linear regression is kept as a sanity-check baseline.

**Validation is chronological** (no shuffling). Shuffling a time-series into
random folds is one of the most common ways to fool yourself with an ML
backtest — it lets the model learn from the future.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from .config import MODEL_DIR, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Summary of a training run — surfaced in the dashboard."""

    reg_rmse: float
    reg_mae: float
    reg_direction_acc: float  # does sign(pred) match sign(actual)?
    clf_accuracy: float
    clf_auc: float
    n_train: int
    n_test: int
    feature_importance: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _time_split(
    X: pd.DataFrame, y: pd.Series, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X)
    split = int(n * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def train_models(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_clf: pd.Series,
    cfg: ModelConfig | None = None,
) -> tuple[XGBRegressor, XGBClassifier, TrainResult]:
    cfg = cfg or ModelConfig()

    Xtr, Xte, yr_tr, yr_te = _time_split(X, y_reg, cfg.test_size)
    _, _, yc_tr, yc_te = _time_split(X, y_clf, cfg.test_size)

    reg = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=cfg.random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    reg.fit(Xtr, yr_tr)
    yr_pred = reg.predict(Xte)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=cfg.random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )
    clf.fit(Xtr, yc_tr)
    yc_prob = clf.predict_proba(Xte)[:, 1]
    yc_pred = (yc_prob > 0.5).astype(int)

    # Feature importance from the classifier — usually more interpretable than
    # the regressor's gain scores because targets are balanced 0/1.
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    result = TrainResult(
        reg_rmse=float(np.sqrt(mean_squared_error(yr_te, yr_pred))),
        reg_mae=float(mean_absolute_error(yr_te, yr_pred)),
        reg_direction_acc=float(np.mean(np.sign(yr_pred) == np.sign(yr_te))),
        clf_accuracy=float(accuracy_score(yc_te, yc_pred)),
        clf_auc=float(roc_auc_score(yc_te, yc_prob)),
        n_train=len(Xtr),
        n_test=len(Xte),
        feature_importance=importances.head(25).to_dict(),
    )
    logger.info(
        "Trained: direction acc %.3f, AUC %.3f, RMSE %.4f",
        result.clf_accuracy,
        result.clf_auc,
        result.reg_rmse,
    )
    return reg, clf, result


def walk_forward_backtest(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_clf: pd.Series,
    cfg: ModelConfig | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Rolling-origin evaluation.

    Splits the series into ``n_splits`` expanding windows. At each step we
    train on everything before the fold and predict on the fold itself. This
    is a much more honest measure of live performance than a single holdout.
    """
    cfg = cfg or ModelConfig()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        yr_tr, yr_te = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        yc_tr, yc_te = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

        reg = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=cfg.random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        reg.fit(Xtr, yr_tr)
        yr_pred = reg.predict(Xte)

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=cfg.random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )
        clf.fit(Xtr, yc_tr)
        yc_prob = clf.predict_proba(Xte)[:, 1]

        records.append(
            {
                "fold": i + 1,
                "train_end": Xtr.index.max(),
                "test_start": Xte.index.min(),
                "test_end": Xte.index.max(),
                "rmse": float(np.sqrt(mean_squared_error(yr_te, yr_pred))),
                "direction_acc": float(np.mean(np.sign(yr_pred) == np.sign(yr_te))),
                "clf_auc": float(roc_auc_score(yc_te, yc_prob)),
            }
        )
    return pd.DataFrame(records)


def simulate_strategy(
    X: pd.DataFrame,
    y_reg: pd.Series,
    cfg: ModelConfig | None = None,
) -> pd.DataFrame:
    """Simple long/flat strategy: go long SPY when predicted return > 0.

    Returns a DataFrame with cumulative equity curves for the strategy vs
    buy-and-hold. Useful sanity check — a classifier with 55% accuracy
    should produce a meaningfully better Sharpe than buy-and-hold.
    """
    cfg = cfg or ModelConfig()
    Xtr, Xte, yr_tr, yr_te = _time_split(X, y_reg, cfg.test_size)

    reg = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        random_state=cfg.random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    reg.fit(Xtr, yr_tr)
    pred = pd.Series(reg.predict(Xte), index=Xte.index, name="pred")

    signal = (pred > 0).astype(int)
    strategy_rets = signal * yr_te
    buyhold_rets = yr_te

    out = pd.DataFrame(
        {
            "pred": pred,
            "actual": yr_te,
            "signal": signal,
            "strategy_equity": strategy_rets.cumsum().apply(np.exp),
            "buyhold_equity": buyhold_rets.cumsum().apply(np.exp),
        }
    )
    return out


def save_models(reg: XGBRegressor, clf: XGBClassifier, result: TrainResult) -> None:
    joblib.dump(reg, MODEL_DIR / "regressor.joblib")
    joblib.dump(clf, MODEL_DIR / "classifier.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))


def load_models() -> tuple[XGBRegressor, XGBClassifier, dict] | None:
    reg_path = MODEL_DIR / "regressor.joblib"
    clf_path = MODEL_DIR / "classifier.joblib"
    metrics_path = MODEL_DIR / "metrics.json"
    if not (reg_path.exists() and clf_path.exists()):
        return None
    reg = joblib.load(reg_path)
    clf = joblib.load(clf_path)
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return reg, clf, metrics


def predict_next(
    reg: XGBRegressor, clf: XGBClassifier, X_latest: pd.DataFrame
) -> dict:
    """Serve a single next-day prediction from the latest feature row."""
    ret_pred = float(reg.predict(X_latest)[0])
    up_prob = float(clf.predict_proba(X_latest)[0, 1])
    return {
        "expected_log_return": ret_pred,
        "expected_pct_return": float(np.expm1(ret_pred)) * 100,
        "prob_up": up_prob,
        "direction": "UP" if up_prob >= 0.5 else "DOWN",
        "confidence": abs(up_prob - 0.5) * 2,  # 0..1
        "as_of": X_latest.index[-1].strftime("%Y-%m-%d"),
    }
