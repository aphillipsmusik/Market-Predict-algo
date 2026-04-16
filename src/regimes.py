"""Unsupervised market-regime detection.

Supervised models predict "what happens tomorrow"; regimes answer a different
question — "what *kind* of market are we in right now?" Prediction accuracy
typically varies a lot by regime, so surfacing the current regime alongside
the point forecast helps the user calibrate how much to trust it.

Approach: compute a handful of rolling statistics (SPY trend, vol, VIX level,
credit spread proxy), then cluster them with K-Means. Cluster centers are
post-hoc labeled as bullish / bearish / high-vol / sideways based on their
statistics, rather than hard-coded — so the labels adapt to the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeModel:
    kmeans: KMeans
    scaler: StandardScaler
    feature_cols: list[str]
    labels: dict[int, str]  # cluster id -> human label


def _regime_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the small feature set used for regime clustering."""
    spy = prices["SPY"]
    rets = np.log(spy / spy.shift(1))

    feats = pd.DataFrame(index=prices.index)
    feats["trend_20"] = spy.pct_change(20)
    feats["trend_60"] = spy.pct_change(60)
    feats["vol_20"] = rets.rolling(20).std()
    feats["drawdown_60"] = spy / spy.rolling(60).max() - 1.0

    if "^VIX" in prices.columns:
        feats["vix_level"] = prices["^VIX"]
        feats["vix_change_20"] = prices["^VIX"].pct_change(20)

    if {"HYG", "IEF"}.issubset(prices.columns):
        feats["credit_spread_proxy"] = (
            np.log(prices["HYG"] / prices["HYG"].shift(20))
            - np.log(prices["IEF"] / prices["IEF"].shift(20))
        )

    return feats.dropna()


def _label_clusters(
    feats: pd.DataFrame, clusters: np.ndarray, k: int
) -> dict[int, str]:
    """Assign human-readable names to each cluster.

    The heuristic ranks clusters by their average 60-day trend + inverse
    volatility, then names them:

      * Highest trend / lowest vol  -> Bull
      * Lowest trend / largest drawdown -> Bear
      * Highest vol / vix spike     -> High Volatility
      * Middle of the pack          -> Sideways / Recovery
    """
    df = feats.copy()
    df["cluster"] = clusters
    stats = df.groupby("cluster").mean(numeric_only=True)

    # Score each cluster: positive trend good, low vol good
    stats["score"] = stats["trend_60"].fillna(0) - stats["vol_20"].fillna(0) * 5
    if "drawdown_60" in stats:
        stats["score"] += stats["drawdown_60"].fillna(0)

    ordered = stats.sort_values("score", ascending=False).index.tolist()

    # Default label pool sized to k
    pool = ["Bull", "Recovery", "Sideways", "Correction", "Bear", "Crisis"]
    labels: dict[int, str] = {}
    for rank, cid in enumerate(ordered):
        labels[int(cid)] = pool[rank] if rank < len(pool) else f"Cluster {cid}"

    # Override: if any cluster has much higher vol than others, mark as
    # "High Volatility" — more useful than "Correction".
    vols = stats["vol_20"]
    if vols.max() > vols.mean() * 1.8:
        labels[int(vols.idxmax())] = "High Volatility"

    return labels


def fit_regimes(prices: pd.DataFrame, k: int = 4, random_state: int = 42) -> RegimeModel:
    feats = _regime_features(prices)
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    clusters = km.fit_predict(X)
    labels = _label_clusters(feats, clusters, k)
    return RegimeModel(kmeans=km, scaler=scaler, feature_cols=list(feats.columns), labels=labels)


def label_history(model: RegimeModel, prices: pd.DataFrame) -> pd.Series:
    """Return a date-indexed series of regime labels for the full history."""
    feats = _regime_features(prices)
    if feats.empty:
        return pd.Series(dtype=object)
    X = model.scaler.transform(feats[model.feature_cols])
    ids = model.kmeans.predict(X)
    return pd.Series([model.labels[int(i)] for i in ids], index=feats.index, name="regime")


def current_regime(model: RegimeModel, prices: pd.DataFrame) -> dict:
    """Current regime + a confidence derived from distance to cluster center."""
    feats = _regime_features(prices)
    if feats.empty:
        return {"regime": "unknown", "confidence": 0.0, "as_of": None}
    x = model.scaler.transform(feats[model.feature_cols].tail(1))
    cid = int(model.kmeans.predict(x)[0])
    # Distance-based confidence: closer to cluster center = more confident
    dists = model.kmeans.transform(x)[0]
    closest = dists[cid]
    second = np.partition(dists, 1)[1]
    # Relative gap between nearest and 2nd-nearest cluster (0..1ish)
    confidence = float((second - closest) / (second + 1e-6))
    return {
        "regime": model.labels[cid],
        "cluster_id": cid,
        "confidence": max(0.0, min(1.0, confidence)),
        "as_of": feats.index[-1].strftime("%Y-%m-%d"),
    }
