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

import joblib  # noqa: E402

from src.config import FEATURE_TICKERS, MODEL_DIR, TARGET_TICKER, ModelConfig  # noqa: E402
from src.data_loader import load_prices  # noqa: E402
from src.ensemble import blend_predictions  # noqa: E402
from src.features import build_feature_matrix, latest_feature_row  # noqa: E402
from src.model import (  # noqa: E402
    load_models,
    predict_next,
    save_models,
    simulate_strategy,
    train_models,
    walk_forward_backtest,
)
from src.regimes import current_regime, fit_regimes, label_history  # noqa: E402

# Deep model (PyTorch) is optional — gracefully degrade if not installed.
try:
    from src.deep_model import (  # noqa: E402
        TORCH_AVAILABLE,
        load_lstm,
        predict_lstm_next,
        save_lstm,
        train_lstm,
    )
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    load_lstm = predict_lstm_next = save_lstm = train_lstm = None  # type: ignore

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


@st.cache_resource
def get_or_train_lstm(prices_signature: str):
    """Load or lazily train the LSTM. Returns (model, artifact) or (None, None)."""
    if not TORCH_AVAILABLE:
        return None, None
    existing_model, existing_artifact = load_lstm()
    if existing_model is not None:
        return existing_model, existing_artifact
    try:
        prices = get_prices("2010-01-01")
        model, artifact, _ = train_lstm(prices, epochs=20)
        save_lstm(artifact)
        return model, artifact
    except Exception:
        return None, None


@st.cache_resource
def get_or_fit_regimes(prices_signature: str):
    path = MODEL_DIR / "regimes.joblib"
    if path.exists():
        return joblib.load(path)
    prices = get_prices("2010-01-01")
    model = fit_regimes(prices, k=4)
    joblib.dump(model, path)
    return model


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
st.sidebar.subheader("Ensemble weights")
w_xgb = st.sidebar.slider("XGBoost weight", 0.0, 1.0, 0.5, 0.05)
w_lstm = 1.0 - w_xgb
st.sidebar.caption(f"LSTM weight: {w_lstm:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Target:** `SPY` (SPDR S&P 500 ETF)\n\n"
    "**Cross-asset inputs:** broad market, VIX, dollar, treasuries, oil, "
    "gold, silver, commodities, sector ETFs, high yield credit.\n\n"
    "**AI models:**\n"
    "- XGBoost (gradient-boosted trees) on engineered features\n"
    "- LSTM (deep learning) on raw cross-asset return sequences\n"
    "- KMeans regime detector (unsupervised)"
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
    with st.spinner("Training XGBoost..."):
        reg, clf, result = train_models(X, y_reg, y_clf)
        save_models(reg, clf, result)
    metrics = result.to_dict()

    if TORCH_AVAILABLE:
        with st.spinner("Training LSTM (deep learning)..."):
            try:
                _, lstm_artifact, _ = train_lstm(prices, epochs=20)
                save_lstm(lstm_artifact)
            except Exception as exc:
                st.warning(f"LSTM training failed: {exc}")

    with st.spinner("Fitting regime detector..."):
        regime_model = fit_regimes(prices, k=4)
        joblib.dump(regime_model, MODEL_DIR / "regimes.joblib")
else:
    reg, clf, metrics = get_or_train_models(str(prices.index.max()))

lstm_model, lstm_artifact = get_or_train_lstm(str(prices.index.max()))
regime_model = get_or_fit_regimes(str(prices.index.max()))

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
# 1. Prediction card (AI ensemble)
# ----------------------------------------------------------------------------- #
st.header("Next-Day Prediction — AI Ensemble")

latest = latest_feature_row(prices)
if latest.empty:
    st.error("Not enough data to build a prediction — try pulling more history.")
    st.stop()

latest_aligned = latest.reindex(columns=X_all.columns, fill_value=np.nan).ffill()
xgb_pred = predict_next(reg, clf, latest_aligned)

# LSTM prediction (optional)
lstm_pred = None
if lstm_model is not None and lstm_artifact is not None:
    try:
        lstm_pred = predict_lstm_next(lstm_model, lstm_artifact, prices)
    except Exception as exc:
        st.warning(f"LSTM prediction failed: {exc}")

ensemble = blend_predictions(xgb_pred, lstm_pred, w_xgb=w_xgb, w_lstm=w_lstm)

# Current market regime
regime_info = current_regime(regime_model, prices)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Direction",
    ensemble["direction"],
    f"{ensemble['prob_up']*100:.1f}% up",
)
col2.metric("Expected return", f"{ensemble['expected_pct_return']:+.3f}%")
col3.metric("Ensemble confidence", f"{ensemble['confidence']*100:.0f}%")
col4.metric("Market regime", regime_info["regime"])
col5.metric("As of", ensemble["as_of"])

last_spy = prices[TARGET_TICKER].iloc[-1]
projected = last_spy * np.exp(ensemble["expected_log_return"])
st.info(
    f"**SPY close on {ensemble['as_of']}:** ${last_spy:,.2f}  →  "
    f"**projected next-day close:** ${projected:,.2f} "
    f"(Δ {ensemble['expected_pct_return']:+.2f}%). "
    f"Regime: **{regime_info['regime']}** "
    f"(regime confidence {regime_info['confidence']*100:.0f}%)."
)

# Per-model breakdown
st.subheader("Per-model breakdown")
mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.markdown("**XGBoost (trees on engineered features)**")
    st.metric("Direction", xgb_pred["direction"], f"{xgb_pred['prob_up']*100:.1f}% up")
    st.metric("Expected", f"{xgb_pred['expected_pct_return']:+.3f}%")
with mcol2:
    st.markdown("**LSTM (deep learning on sequences)**")
    if lstm_pred is not None:
        st.metric(
            "Direction", lstm_pred["direction"], f"{lstm_pred['prob_up']*100:.1f}% up"
        )
        st.metric("Expected", f"{lstm_pred['expected_pct_return']:+.3f}%")
    else:
        if not TORCH_AVAILABLE:
            st.caption("PyTorch not installed. `pip install torch` to enable.")
        else:
            st.caption("LSTM not trained yet. Click **Retrain models**.")
with mcol3:
    st.markdown("**Ensemble (weighted blend)**")
    st.metric(
        "Direction", ensemble["direction"], f"{ensemble['prob_up']*100:.1f}% up"
    )
    st.metric("Expected", f"{ensemble['expected_pct_return']:+.3f}%")
st.caption(
    "This is a statistical point estimate, not a guarantee. The 1-day return "
    "standard deviation for SPY is ~1%, so treat single-day predictions as "
    "direction-biased expected values, not precise targets."
)

# ----------------------------------------------------------------------------- #
# 2. AI Pattern / Regime panel
# ----------------------------------------------------------------------------- #
st.header("AI Pattern & Regime Detection")
st.caption(
    "Unsupervised K-Means clustering on rolling trend, volatility, drawdown, "
    "and credit-spread features identifies the current *market regime*. "
    "Model accuracy tends to vary by regime — this helps calibrate trust."
)

regime_series = label_history(regime_model, prices)
spy_hist = prices[TARGET_TICKER].reindex(regime_series.index)

# Color-code SPY by regime
regime_palette = {
    "Bull": "#2ca02c",
    "Recovery": "#98df8a",
    "Sideways": "#7f7f7f",
    "Correction": "#ff9896",
    "Bear": "#d62728",
    "Crisis": "#8b0000",
    "High Volatility": "#ff7f0e",
}

regime_fig = go.Figure()
regime_fig.add_trace(
    go.Scatter(
        x=spy_hist.index,
        y=spy_hist.values,
        mode="lines",
        line=dict(color="#1f77b4", width=1),
        name="SPY",
    )
)
# Overlay regime as colored markers
for regime_name in regime_series.unique():
    mask = regime_series == regime_name
    regime_fig.add_trace(
        go.Scatter(
            x=spy_hist.index[mask],
            y=spy_hist.values[mask],
            mode="markers",
            marker=dict(size=4, color=regime_palette.get(regime_name, "#999")),
            name=regime_name,
        )
    )
regime_fig.update_layout(
    title="SPY price history colored by detected regime",
    height=420,
    yaxis_title="SPY ($)",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(regime_fig, use_container_width=True)

# Regime distribution / stats
rc1, rc2 = st.columns([1, 2])
with rc1:
    regime_counts = regime_series.value_counts()
    pie = go.Figure(
        go.Pie(
            labels=regime_counts.index,
            values=regime_counts.values,
            marker=dict(
                colors=[regime_palette.get(n, "#999") for n in regime_counts.index]
            ),
        )
    )
    pie.update_layout(title="Historical regime distribution", height=350)
    st.plotly_chart(pie, use_container_width=True)

with rc2:
    # Average forward return conditional on regime — useful for calibration
    fwd = np.log(prices[TARGET_TICKER] / prices[TARGET_TICKER].shift(1)).shift(-1)
    fwd = fwd.reindex(regime_series.index)
    stats = (
        pd.DataFrame({"regime": regime_series, "fwd_return": fwd})
        .groupby("regime")["fwd_return"]
        .agg(
            mean_pct=lambda s: s.mean() * 100,
            vol_pct=lambda s: s.std() * 100,
            up_rate=lambda s: (s > 0).mean() * 100,
            n_days="count",
        )
        .sort_values("mean_pct", ascending=False)
    )
    st.markdown("**Next-day return statistics by regime**")
    st.dataframe(
        stats.style.format(
            {
                "mean_pct": "{:+.3f}%",
                "vol_pct": "{:.3f}%",
                "up_rate": "{:.1f}%",
                "n_days": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

# ----------------------------------------------------------------------------- #
# 3. Price explorer
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
# 4. Model diagnostics
# ----------------------------------------------------------------------------- #
st.header("Model Diagnostics")

st.subheader("XGBoost (gradient-boosted trees)")
dcol1, dcol2, dcol3, dcol4 = st.columns(4)
dcol1.metric("Direction accuracy", f"{metrics.get('clf_accuracy', 0)*100:.1f}%")
dcol2.metric("ROC-AUC", f"{metrics.get('clf_auc', 0):.3f}")
dcol3.metric("RMSE (log return)", f"{metrics.get('reg_rmse', 0):.4f}")
dcol4.metric("Regressor sign acc.", f"{metrics.get('reg_direction_acc', 0)*100:.1f}%")

if lstm_artifact is not None:
    st.subheader("LSTM (deep learning)")
    lm = lstm_artifact.metrics
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Direction accuracy", f"{lm.get('clf_accuracy', 0)*100:.1f}%")
    lc2.metric("ROC-AUC", f"{lm.get('clf_auc', 0):.3f}")
    lc3.metric("RMSE (log return)", f"{lm.get('reg_rmse', 0):.4f}")
    lc4.metric("Regressor sign acc.", f"{lm.get('reg_direction_acc', 0)*100:.1f}%")
    st.caption(
        f"Trained for {int(lm.get('epochs', 0))} epochs on "
        f"{int(lm.get('n_train', 0))} sequences of {lstm_artifact.seq_len} "
        f"trading days × {len(lstm_artifact.feature_columns)} tickers."
    )

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
        title="Top 20 feature importances (XGBoost classifier)",
    )
    fig_fi.update_layout(height=600)
    st.plotly_chart(fig_fi, use_container_width=True)

# ----------------------------------------------------------------------------- #
# 5. Backtest
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
