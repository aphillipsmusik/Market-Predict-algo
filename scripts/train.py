"""Train the SPY prediction models and persist them to ``models/``.

Run from the repo root:

    python -m scripts.train [--refresh] [--start 2010-01-01]
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import DEFAULT_START, ModelConfig
from src.data_loader import load_prices
from src.features import build_feature_matrix
from src.model import save_models, train_models, walk_forward_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SPY cross-asset models.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--refresh", action="store_true", help="Re-download prices.")
    parser.add_argument(
        "--no-backtest", action="store_true", help="Skip walk-forward CV."
    )
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

    reg, clf, result = train_models(X, y_reg, y_clf, cfg)
    save_models(reg, clf, result)
    print("\n=== Holdout metrics ===")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
