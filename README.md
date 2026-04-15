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

## AI stack

Three complementary models work together, and the dashboard shows each one
plus their ensemble:

| Model | Type | What it captures |
| ----- | ---- | ---------------- |
| **XGBoost** (regressor + classifier) | Gradient-boosted trees | Nonlinear relationships between *engineered* cross-asset features (lags, MAs, RSI, spreads) |
| **LSTM** (`src/deep_model.py`) | Deep learning / recurrent neural net | *Temporal patterns* directly from raw return sequences — what XGBoost can't see because it treats each day independently |
| **KMeans regime detector** (`src/regimes.py`) | Unsupervised clustering | The current *market regime* (Bull / Sideways / Correction / High Volatility / Bear) based on rolling trend, vol, drawdown, credit spread |
| **Ensemble** (`src/ensemble.py`) | Weighted blend | Averages XGB + LSTM — different models make different mistakes, so blending reduces variance |

The regime label is shown alongside every prediction because *model accuracy
varies by regime* — e.g. trend-following signals are better in bull markets,
mean-reversion signals better in sideways ones.

## Project layout

```
Market-Predict-algo/
├── requirements.txt
├── src/
│   ├── config.py         # Ticker universe + hyperparameters
│   ├── data_loader.py    # yfinance download + parquet cache
│   ├── features.py       # Lags, returns, MAs, RSI, vol, cross-asset spreads
│   ├── model.py          # XGBoost regressor + classifier, walk-forward CV,
│   │                     #   long/flat backtest simulator
│   ├── deep_model.py     # PyTorch LSTM sequence model
│   ├── regimes.py        # KMeans market-regime detector (unsupervised)
│   └── ensemble.py       # Blend XGBoost + LSTM predictions
├── scripts/
│   └── train.py          # CLI: train + save all models
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
python -m scripts.train --no-lstm               # Skip LSTM (faster)
python -m scripts.train --no-regimes            # Skip regime detector
python -m scripts.train --epochs 50             # Longer LSTM training
```

---

## What the dashboard shows

1. **AI ensemble prediction** — direction, probability, expected % return,
   projected SPY close, **current market regime**, plus a per-model breakdown
   (XGBoost vs LSTM vs blended).
2. **AI pattern & regime detection** — SPY price colored by detected regime,
   historical regime distribution, and next-day return statistics *conditional
   on regime* (so you can see e.g. "average next-day return in Bull regime is
   +0.08% with 57% up-rate" vs "in High Volatility it's -0.02% with 49%").
3. **Price & correlation explorer** — normalized price overlay plus a
   rolling-correlation bar chart showing which cross-asset tickers are
   currently most linked to SPY.
4. **Model diagnostics** — holdout accuracy, ROC-AUC, and RMSE for *both*
   the XGBoost and LSTM, plus a top-20 feature-importance chart.
5. **Backtest** —
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
- **XGBoost regressor:** targets SPY's next-day log return.
- **XGBoost classifier:** targets up/down direction (probability).
- **LSTM:** 2-layer recurrent net (hidden=64, 30-day lookback, ~17 input
  features). Two output heads — regression + classification — trained jointly
  with MSE + BCE loss. CPU training takes ~30–90 seconds on modern hardware.
- **Regime detector:** `KMeans(k=4)` on standardized rolling trend /
  volatility / drawdown / credit-spread features. Cluster labels (Bull,
  Recovery, Sideways, Correction, Bear, High Volatility) are assigned
  automatically by ranking clusters on a trend-minus-vol score.
- **Ensemble:** weighted average of XGBoost + LSTM probabilities and expected
  returns. Weights adjustable from the dashboard sidebar.

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
