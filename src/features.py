"""Feature engineering for the SPY prediction model.

Key design choices:
  * Features are computed from *returns*, not raw prices, so they're stationary
    and comparable across tickers with very different price scales.
  * All features at time ``t`` use data from ``t`` or earlier. Nothing from the
    future leaks in, which would make backtest performance fraudulent.
  * The target is SPY's return from ``t`` to ``t+horizon``, shifted back so
    that row ``t`` contains ``(features_t, target_t_to_t+h)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET_TICKER, ModelConfig


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns are nicer for ML (more Gaussian, additive over time)."""
    return np.log(prices / prices.shift(1))


def _rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder's RSI on a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_feature_matrix(
    prices: pd.DataFrame,
    cfg: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build ``(X, y_reg, y_clf)`` from a wide price panel.

    Returns
    -------
    X : DataFrame
        Feature matrix. One row per trading day.
    y_reg : Series
        Regression target — SPY log return ``h`` days ahead.
    y_clf : Series
        Classification target — 1 if ``y_reg`` > threshold else 0.
    """
    cfg = cfg or ModelConfig()
    if TARGET_TICKER not in prices.columns:
        raise ValueError(f"Target {TARGET_TICKER} missing from price panel")

    rets = _daily_returns(prices)
    feats: dict[str, pd.Series] = {}

    for ticker in prices.columns:
        r = rets[ticker]
        p = prices[ticker]

        # Lagged returns — captures short-term momentum / reversal
        for lag in cfg.lookback_lags:
            feats[f"{ticker}_ret_lag{lag}"] = r.shift(lag)

        # Moving-average distance — is the ticker above or below its trend?
        for w in cfg.ma_windows:
            ma = p.rolling(w).mean()
            feats[f"{ticker}_ma_dist_{w}"] = (p / ma - 1.0).shift(1)

        # Realized volatility — risk regime proxy
        for w in cfg.vol_windows:
            feats[f"{ticker}_vol_{w}"] = _rolling_vol(r, w).shift(1)

        # RSI — mean-reversion / overbought signal
        feats[f"{ticker}_rsi14"] = _rsi(p, 14).shift(1)

    # Cross-asset ratios and spreads that often matter for equities.
    # All shifted by 1 so we only use prior-day info.
    if {"SPY", "TLT"}.issubset(prices.columns):
        feats["SPY_TLT_ratio"] = (prices["SPY"] / prices["TLT"]).pct_change().shift(1)
    if {"XLE", "SPY"}.issubset(prices.columns):
        feats["XLE_SPY_spread"] = (rets["XLE"] - rets["SPY"]).shift(1)
    if {"HYG", "IEF"}.issubset(prices.columns):
        # Credit spread proxy: high-yield vs safe bonds
        feats["credit_proxy"] = (rets["HYG"] - rets["IEF"]).shift(1)
    if "^VIX" in prices.columns:
        feats["VIX_level"] = prices["^VIX"].shift(1)
        feats["VIX_change_5d"] = prices["^VIX"].pct_change(5).shift(1)

    # Calendar features — day-of-week effects are modest but real.
    X = pd.DataFrame(feats, index=prices.index)
    X["dow"] = X.index.dayofweek
    X["month"] = X.index.month

    # Build targets from the TARGET_TICKER
    spy_fwd = rets[TARGET_TICKER].shift(-cfg.target_horizon)
    y_reg = spy_fwd.rename("spy_fwd_return")
    y_clf = (y_reg > cfg.direction_threshold).astype(int).rename("spy_up")

    # Drop any row where features or target are missing. We do this *once*
    # at the end so the alignment stays clean.
    df = pd.concat([X, y_reg, y_clf], axis=1).dropna()
    X_clean = df.drop(columns=["spy_fwd_return", "spy_up"])
    return X_clean, df["spy_fwd_return"], df["spy_up"]


def latest_feature_row(prices: pd.DataFrame, cfg: ModelConfig | None = None) -> pd.DataFrame:
    """Return the feature vector for the most recent date (target will be NaN).

    Used when serving a live prediction from the dashboard.
    """
    cfg = cfg or ModelConfig()
    rets = _daily_returns(prices)
    feats: dict[str, pd.Series] = {}

    for ticker in prices.columns:
        r = rets[ticker]
        p = prices[ticker]
        for lag in cfg.lookback_lags:
            feats[f"{ticker}_ret_lag{lag}"] = r.shift(lag)
        for w in cfg.ma_windows:
            ma = p.rolling(w).mean()
            feats[f"{ticker}_ma_dist_{w}"] = (p / ma - 1.0).shift(1)
        for w in cfg.vol_windows:
            feats[f"{ticker}_vol_{w}"] = _rolling_vol(r, w).shift(1)
        feats[f"{ticker}_rsi14"] = _rsi(p, 14).shift(1)

    if {"SPY", "TLT"}.issubset(prices.columns):
        feats["SPY_TLT_ratio"] = (prices["SPY"] / prices["TLT"]).pct_change().shift(1)
    if {"XLE", "SPY"}.issubset(prices.columns):
        feats["XLE_SPY_spread"] = (rets["XLE"] - rets["SPY"]).shift(1)
    if {"HYG", "IEF"}.issubset(prices.columns):
        feats["credit_proxy"] = (rets["HYG"] - rets["IEF"]).shift(1)
    if "^VIX" in prices.columns:
        feats["VIX_level"] = prices["^VIX"].shift(1)
        feats["VIX_change_5d"] = prices["^VIX"].pct_change(5).shift(1)

    X = pd.DataFrame(feats, index=prices.index)
    X["dow"] = X.index.dayofweek
    X["month"] = X.index.month
    return X.dropna().tail(1)
