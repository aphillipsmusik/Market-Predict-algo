"""Streamlit dashboard for the SPY prediction algorithm.

Run with:

    streamlit run app/dashboard.py

The dashboard has four sections:

  1. **Prediction card** — today's expected next-day return + direction prob.
  2. **Price explorer** — SPY overlaid with selected cross-asset tickers,
     plus a rolling-correlation heatmap so you can see which assets are
     currently most linked to SPY.
  3. **Model diagnostics** — holdout metrics + top feature importances.
  4. **Backtest** — walk-forward CV scores and a simple long/flat equity
     curve vs buy-and-hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make ``src`` importable when launched via ``streamlit run app/dashboard.py``
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_TICKERS, TARGET_TICKER, ModelConfig  # noqa: E402
from src.data_loader import load_prices  # noqa: E402
from src.features import build_feature_matrix, latest_feature_row  # noqa: E402
from src.model import (  # noqa: E402
    load_models,
    predict_next,
    save_models,
    simulate_strategy,
    train_models,
    walk_forward_backtest,
)

st.set_page_config(
    page_title="SPY Cross-Asset Predictor",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


# ----------------------------------------------------------------------------- #
# Cached loaders
# ----------------------------------------------------------------------------- #
@st.cache_data(ttl=60 * 60)
def get_prices(start: str, refresh: bool = False) -> pd.DataFrame:
    return load_prices(start=start, refresh=refresh)


@st.cache_data(ttl=60 * 60)
def get_features(prices: pd.DataFrame):
    cfg = ModelConfig()
    return build_feature_matrix(prices, cfg)


@st.cache_resource
def get_or_train_models(prices_signature: str):
    """Load models if already trained; otherwise train and persist them."""
    existing = load_models()
    if existing is not None:
        reg, clf, metrics = existing
        return reg, clf, metrics
    prices = get_prices("2010-01-01")
    X, y_reg, y_clf = get_features(prices)
    reg, clf, result = train_models(X, y_reg, y_clf)
    save_models(reg, clf, result)
    return reg, clf, result.to_dict()


# ----------------------------------------------------------------------------- #
# Sidebar
# ----------------------------------------------------------------------------- #
st.sidebar.title("Controls")
start_date = st.sidebar.date_input(
    "Training data start",
    value=pd.Timestamp("2010-01-01"),
    min_value=pd.Timestamp("2005-01-01"),
).strftime("%Y-%m-%d")
refresh = st.sidebar.button("Refresh prices from Yahoo")
retrain = st.sidebar.button("Retrain models")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Target:** `SPY` (SPDR S&P 500 ETF)\n\n"
    "**Cross-asset inputs:** broad market, VIX, dollar, treasuries, oil, "
    "gold, silver, commodities, sector ETFs, high yield credit."
)

# ----------------------------------------------------------------------------- #
# Load data + models
# ----------------------------------------------------------------------------- #
with st.spinner("Loading price history..."):
    prices = get_prices(start_date, refresh=refresh)

if retrain:
    # Blow away cached models, rebuild from scratch
    st.cache_resource.clear()
    X, y_reg, y_clf = get_features(prices)
    with st.spinner("Training models..."):
        reg, clf, result = train_models(X, y_reg, y_clf)
        save_models(reg, clf, result)
    metrics = result.to_dict()
else:
    reg, clf, metrics = get_or_train_models(str(prices.index.max()))

X_all, y_reg_all, y_clf_all = get_features(prices)

# ----------------------------------------------------------------------------- #
# Header
# ----------------------------------------------------------------------------- #
st.title("SPY Cross-Asset Prediction Dashboard")
st.caption(
    "Predicting next-day SPY direction using oil, energy, the dollar, bonds, "
    "volatility, and sector ETFs. Trained on historical data via yfinance."
)

# ----------------------------------------------------------------------------- #
# 1. Prediction card
# ----------------------------------------------------------------------------- #
st.header("Next-Day Prediction")

latest = latest_feature_row(prices)
if latest.empty:
    st.error("Not enough data to build a prediction — try pulling more history.")
    st.stop()

latest_aligned = latest.reindex(columns=X_all.columns, fill_value=np.nan).ffill()
pred = predict_next(reg, clf, latest_aligned)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Direction", pred["direction"], f"{pred['prob_up']*100:.1f}% up")
col2.metric("Expected return", f"{pred['expected_pct_return']:+.3f}%")
col3.metric("Confidence", f"{pred['confidence']*100:.0f}%")
col4.metric("As of", pred["as_of"])

last_spy = prices[TARGET_TICKER].iloc[-1]
projected = last_spy * np.exp(pred["expected_log_return"])
st.info(
    f"**SPY close on {pred['as_of']}:** ${last_spy:,.2f}  →  "
    f"**projected next-day close:** ${projected:,.2f} "
    f"(Δ {pred['expected_pct_return']:+.2f}%)."
)
st.caption(
    "This is a statistical point estimate, not a guarantee. The 1-day return "
    "standard deviation for SPY is ~1%, so treat single-day predictions as "
    "direction-biased expected values, not precise targets."
)

# ----------------------------------------------------------------------------- #
# 2. Price explorer
# ----------------------------------------------------------------------------- #
st.header("Price & Correlation Explorer")

selected = st.multiselect(
    "Overlay tickers (normalized to 100 at window start)",
    options=list(FEATURE_TICKERS.keys()),
    default=["QQQ", "DXY", "USO", "VIX"],
)

window_days = st.slider("Lookback window (trading days)", 60, 2000, 500, step=20)
window = prices.tail(window_days)

label_to_sym = {**{TARGET_TICKER: TARGET_TICKER}, **FEATURE_TICKERS}
plot_symbols = [TARGET_TICKER] + [label_to_sym[k] for k in selected if k in label_to_sym]
plot_symbols = [s for s in plot_symbols if s in window.columns]

normalized = window[plot_symbols].div(window[plot_symbols].iloc[0]).mul(100)
fig = px.line(
    normalized,
    title=f"Normalized prices — SPY vs selected ({window_days} days)",
    labels={"value": "Indexed to 100", "index": "Date", "variable": "Ticker"},
)
fig.update_layout(height=450, legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap — of daily returns over the selected window.
rets = np.log(window / window.shift(1)).dropna()
rename_map = {v: k for k, v in FEATURE_TICKERS.items()}
rets_display = rets.rename(columns=rename_map)
corr = rets_display.corr()

if TARGET_TICKER in corr.columns:
    spy_corr = corr[TARGET_TICKER].drop(TARGET_TICKER).sort_values()
    bar = go.Figure(
        go.Bar(
            x=spy_corr.values,
            y=spy_corr.index,
            orientation="h",
            marker_color=["#d62728" if v < 0 else "#2ca02c" for v in spy_corr.values],
        )
    )
    bar.update_layout(
        title=f"Correlation with SPY (last {window_days} days of daily returns)",
        xaxis_title="Pearson correlation",
        height=500,
    )
    st.plotly_chart(bar, use_container_width=True)

with st.expander("Full correlation matrix"):
    st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))

# ----------------------------------------------------------------------------- #
# 3. Model diagnostics
# ----------------------------------------------------------------------------- #
st.header("Model Diagnostics")

dcol1, dcol2, dcol3, dcol4 = st.columns(4)
dcol1.metric("Direction accuracy", f"{metrics.get('clf_accuracy', 0)*100:.1f}%")
dcol2.metric("ROC-AUC", f"{metrics.get('clf_auc', 0):.3f}")
dcol3.metric("RMSE (log return)", f"{metrics.get('reg_rmse', 0):.4f}")
dcol4.metric("Regressor sign acc.", f"{metrics.get('reg_direction_acc', 0)*100:.1f}%")

fi = metrics.get("feature_importance", {})
if fi:
    fi_df = (
        pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
        .sort_values("importance", ascending=True)
        .tail(20)
    )
    fig_fi = px.bar(
        fi_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 20 feature importances (classifier)",
    )
    fig_fi.update_layout(height=600)
    st.plotly_chart(fig_fi, use_container_width=True)

# ----------------------------------------------------------------------------- #
# 4. Backtest
# ----------------------------------------------------------------------------- #
st.header("Backtest")
tab_wf, tab_eq = st.tabs(["Walk-forward CV", "Long/flat equity curve"])

with tab_wf:
    st.caption(
        "Expanding-window time-series cross validation. Each fold trains on "
        "everything before its test period — no future data leaks in."
    )
    with st.spinner("Running walk-forward CV..."):
        bt = walk_forward_backtest(X_all, y_reg_all, y_clf_all)
    st.dataframe(bt, use_container_width=True)
    st.write(
        f"**Mean direction accuracy:** {bt['direction_acc'].mean()*100:.2f}%  |  "
        f"**Mean AUC:** {bt['clf_auc'].mean():.3f}"
    )

with tab_eq:
    st.caption(
        "Go long SPY when the model predicts positive next-day return, "
        "otherwise stay in cash. Purely illustrative — no transaction costs."
    )
    sim = simulate_strategy(X_all, y_reg_all)
    eq_fig = go.Figure()
    eq_fig.add_trace(
        go.Scatter(x=sim.index, y=sim["strategy_equity"], name="Model long/flat")
    )
    eq_fig.add_trace(
        go.Scatter(x=sim.index, y=sim["buyhold_equity"], name="Buy & hold SPY")
    )
    eq_fig.update_layout(
        title="Cumulative growth of $1 (test period)",
        yaxis_title="Equity multiple",
        height=450,
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    strat_ret = sim["strategy_equity"].iloc[-1] - 1
    bh_ret = sim["buyhold_equity"].iloc[-1] - 1
    c1, c2 = st.columns(2)
    c1.metric("Model total return", f"{strat_ret*100:+.1f}%")
    c2.metric("Buy & hold total return", f"{bh_ret*100:+.1f}%")

st.markdown("---")
st.caption(
    "Educational / research tool. Not investment advice. Markets are noisy and "
    "no model is stable forever — retrain regularly and stress-test before "
    "using for real capital allocation."
)
