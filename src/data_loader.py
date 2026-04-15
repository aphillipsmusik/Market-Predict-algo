"""Download and cache historical OHLCV data from Yahoo Finance.

We use yfinance because it's free and requires no API key. Data is cached to
Parquet so that repeated runs (and the dashboard) don't re-hit the network.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Iterable

import pandas as pd
import yfinance as yf

from .config import ALL_TICKERS, CACHE_FILE, DEFAULT_START

logger = logging.getLogger(__name__)


def _download(tickers: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers.

    Returns a wide DataFrame indexed by date with one column per ticker.
    """
    tickers = list(tickers)
    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance returns a MultiIndex when we pass multiple tickers. We want a
    # flat DataFrame of closing prices (auto_adjust=True means Close is already
    # split/dividend adjusted).
    if isinstance(raw.columns, pd.MultiIndex):
        close = pd.DataFrame(
            {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)}
        )
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    return close


def load_prices(
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load a price panel, using the local parquet cache when possible.

    Parameters
    ----------
    start, end : str | None
        ISO date strings. ``end=None`` means today.
    refresh : bool
        If True, ignore the cache and re-download.
    tickers : iterable, optional
        Subset of tickers. Defaults to every ticker in ``config.ALL_TICKERS``.
    """
    tickers = list(tickers) if tickers is not None else ALL_TICKERS
    end = end or datetime.utcnow().strftime("%Y-%m-%d")

    if CACHE_FILE.exists() and not refresh:
        cached = pd.read_parquet(CACHE_FILE)
        cached.index = pd.to_datetime(cached.index)
        have_tickers = set(cached.columns)
        need_tickers = set(tickers)
        cache_start = cached.index.min().strftime("%Y-%m-%d")
        cache_end = cached.index.max().strftime("%Y-%m-%d")

        if need_tickers.issubset(have_tickers) and cache_start <= start and cache_end >= end[:10]:
            return cached.loc[start:end, tickers]

        # Otherwise fall through and re-download. Covering all edge cases
        # (merging partial caches) adds complexity without much upside.
        logger.info("Cache miss / stale, refreshing.")

    prices = _download(tickers, start=start, end=end)
    prices.to_parquet(CACHE_FILE)
    return prices


def last_available_date(prices: pd.DataFrame) -> date:
    return prices.index.max().date()
