"""Train the SPY prediction models and persist them to ``models/``.

Run from the repo root:

    python -m scripts.train [--refresh] [--start 2010-01-01] [--no-lstm]

By default this trains three things:
  1. XGBoost regressor + classifier on engineered cross-asset features
  2. LSTM sequence model on raw cross-asset returns (deep learning)
  3. KMeans regime detector for the "what kind of market are we in?" label
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import joblib

from src.config import DEFAULT_START, MODEL_DIR, ModelConfig
from src.data_loader import load_prices
from src.features import build_feature_matrix
from src.model import save_models, train_models, walk_forward_backtest
from src.regimes import fit_regimes


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SPY cross-asset models.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--refresh", action="store_true", help="Re-download prices.")
    parser.add_argument("--no-backtest", action="store_true", help="Skip walk-forward CV.")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM training.")
    parser.add_argument("--no-regimes", action="store_true", help="Skip regime detection.")
    parser.add_argument("--epochs", type=int, default=25, help="LSTM training epochs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s"
    )

    prices = load_prices(start=args.start, end=args.end, refresh=args.refresh)
    print(f"Loaded prices: {prices.shape[0]} rows x {prices.shape[1]} tickers")
    print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")

    cfg = ModelConfig()
    X, y_reg, y_clf = build_feature_matrix(prices, cfg)
    print(f"Feature matrix: {X.shape}")

    # ------------------------------------------------------------------ XGB
    reg, clf, result = train_models(X, y_reg, y_clf, cfg)
    save_models(reg, clf, result)
    print("\n=== XGBoost holdout metrics ===")
    print(f"  Regression RMSE      : {result.reg_rmse:.5f}")
    print(f"  Regression MAE       : {result.reg_mae:.5f}")
    print(f"  Regression dir. acc  : {result.reg_direction_acc:.3f}")
    print(f"  Classifier accuracy  : {result.clf_accuracy:.3f}")
    print(f"  Classifier ROC-AUC   : {result.clf_auc:.3f}")
    print(f"  Train / Test rows    : {result.n_train} / {result.n_test}")

    print("\n=== Top 10 features ===")
    for name, imp in list(result.feature_importance.items())[:10]:
        print(f"  {name:35s}  {imp:.4f}")

    if not args.no_backtest:
        print("\n=== Walk-forward backtest ===")
        bt = walk_forward_backtest(X, y_reg, y_clf, cfg)
        print(bt.to_string(index=False))

    # ------------------------------------------------------------------ LSTM
    if not args.no_lstm:
        try:
            from src.deep_model import TORCH_AVAILABLE, save_lstm, train_lstm

            if not TORCH_AVAILABLE:
                print("\n[skip] PyTorch not installed — run `pip install torch` for LSTM.")
            else:
                print("\n=== Training LSTM (this takes ~30-90s on CPU) ===")
                _, artifact, lstm_result = train_lstm(
                    prices, cfg=cfg, epochs=args.epochs
                )
                save_lstm(artifact)
                print(f"  LSTM direction acc   : {lstm_result.clf_accuracy:.3f}")
                print(f"  LSTM ROC-AUC         : {lstm_result.clf_auc:.3f}")
                print(f"  LSTM regr. RMSE      : {lstm_result.reg_rmse:.5f}")
                print(f"  LSTM regr. dir. acc  : {lstm_result.reg_direction_acc:.3f}")
                print(f"  Train / Test rows    : {lstm_result.n_train} / {lstm_result.n_test}")
        except Exception as exc:  # don't let LSTM failure kill the whole run
            print(f"\n[warn] LSTM training failed: {exc}")

    # ------------------------------------------------------------------ Regimes
    if not args.no_regimes:
        print("\n=== Fitting market regime detector (KMeans) ===")
        regime_model = fit_regimes(prices, k=4)
        joblib.dump(regime_model, MODEL_DIR / "regimes.joblib")
        (MODEL_DIR / "regimes_labels.json").write_text(
            json.dumps({str(k): v for k, v in regime_model.labels.items()}, indent=2)
        )
        print("  Regimes learned:")
        for cid, name in regime_model.labels.items():
            print(f"    cluster {cid}: {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
