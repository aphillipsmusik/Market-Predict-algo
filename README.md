# Market-Predict-algo

A research project that predicts **SPY** (SPDR S&P 500 ETF) next-day direction
and expected return using cross-asset signals — oil, energy, the US dollar,
treasuries, gold, volatility, sector ETFs, and credit.

Includes a training pipeline, walk-forward backtest, and a **Streamlit
dashboard** for exploration and live predictions.

> Educational / research tool — **not investment advice**.

---

## Can you actually predict SPY from other tickers?

Partially. SPY has well-documented statistical relationships with:

| Asset              | Typical relationship                                           |
| ------------------ | -------------------------------------------------------------- |
| `^VIX`             | Strongly negative — fear up, stocks down                       |
| `UUP` / dollar     | Usually mildly negative — strong dollar weighs on multinationals |
| `TLT` / long bonds | Regime dependent — flight-to-safety vs. rate fear              |
| `USO` / oil        | Mixed — supply shocks hurt, demand strength helps              |
| `XLE` / energy     | Leading indicator in inflation regimes                         |
| `HYG` / high yield | Positive — credit spreads widen before equity drawdowns        |

No model can predict the *exact* closing price. A realistic, honest goal is
**next-day direction** with 53–58% accuracy, which — if stable — is
statistically meaningful given how noisy markets are.

---

## Project layout

```
Market-Predict-algo/
├── requirements.txt
├── src/
│   ├── config.py         # Ticker universe + hyperparameters
│   ├── data_loader.py    # yfinance download + parquet cache
│   ├── features.py       # Lags, returns, MAs, RSI, vol, cross-asset spreads
│   └── model.py          # XGBoost regressor + classifier, walk-forward CV,
│                         #   long/flat backtest simulator
├── scripts/
│   └── train.py          # CLI: train + save models
├── app/
│   └── dashboard.py      # Streamlit dashboard
├── data/                 # Cached prices (gitignored)
└── models/               # Saved models + metrics (gitignored)
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train (downloads ~15 tickers from 2010 → today, builds features, trains)
python -m scripts.train

# 2. Launch the dashboard
streamlit run app/dashboard.py
```

The first run downloads price history (a few seconds) and caches it in
`data/prices.parquet`. Trained models go in `models/`.

### Useful flags

```bash
python -m scripts.train --refresh               # Re-download prices
python -m scripts.train --start 2015-01-01      # Train on a shorter window
python -m scripts.train --no-backtest           # Skip walk-forward CV
```

---

## What the dashboard shows

1. **Next-day prediction** — direction, probability, expected % return, and a
   projected SPY close.
2. **Price & correlation explorer** — normalized price overlay plus a
   rolling-correlation bar chart showing which cross-asset tickers are
   currently most linked to SPY.
3. **Model diagnostics** — holdout accuracy, ROC-AUC, RMSE, and a top-20
   feature-importance chart so you can see which signals the model leans on.
4. **Backtest** —
   - Walk-forward cross-validation (5 expanding folds, no future leakage).
   - Long/flat equity curve vs buy-and-hold on the holdout period.

---

## How it works

### Features
All features are built from **prior-day or earlier** data — no look-ahead.

- **Per-ticker:** lagged log returns (1, 2, 3, 5, 10, 20 day), moving-average
  distance (5, 10, 20, 50), rolling volatility (10, 20), and 14-day RSI.
- **Cross-asset:** SPY/TLT ratio, XLE–SPY return spread, HYG–IEF credit proxy,
  VIX level and 5-day change.
- **Calendar:** day-of-week, month.

### Models
- **Regressor:** `XGBRegressor` targeting SPY's next-day log return.
- **Classifier:** `XGBClassifier` targeting up/down direction (probability).
- **Baseline:** hyperparameters chosen for stability, not bleeding-edge
  accuracy. Feel free to tune `ModelConfig` in `src/config.py`.

### Validation
- Chronological 80/20 holdout for the saved model's headline metrics.
- 5-fold `TimeSeriesSplit` walk-forward CV for honest out-of-sample estimates.
- Long/flat strategy simulation on the holdout as a sanity check — a useful
  model should beat buy-and-hold on risk-adjusted basis even ignoring costs.

---

## Caveats (read these)

- Markets are **non-stationary**. A model trained through 2019 knew nothing
  about COVID; one trained through 2021 didn't know about the 2022 rate
  shock. Retrain regularly.
- yfinance data is free but not clean enough for production. For serious use,
  swap in a paid data feed (Polygon, Tiingo, or similar).
- No transaction costs, slippage, borrowing costs, or taxes are modeled.
- Single-day predictions have inherently wide uncertainty (SPY's 1-day
  standard deviation is ~1%). Trust the *direction and probability*, not the
  point estimate.
