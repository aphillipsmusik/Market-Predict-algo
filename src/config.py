"""Central configuration for tickers, date ranges, and model parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Target we want to predict
TARGET_TICKER = "SPY"

# Cross-asset features. These are the "other important tickers" the algorithm
# uses to help predict SPY direction / next-day returns.
#
# Each entry maps a human-friendly label -> Yahoo Finance ticker symbol.
# Chosen to span the main macro drivers of equity risk premium:
#   - Broad market internals (QQQ, IWM)
#   - Volatility / fear (^VIX)
#   - Dollar (UUP ETF is more reliable on yfinance than DX-Y.NYB)
#   - Rates / bonds (TLT long-duration, IEF intermediate, ^TNX 10y yield)
#   - Commodities (USO oil, GLD gold, SLV silver, DBC broad commodities)
#   - Sectors that often lead (XLE energy, XLF financials, XLK tech, XLU utilities)
#   - Credit / risk-on-off (HYG high yield)
FEATURE_TICKERS: dict[str, str] = {
    "QQQ": "QQQ",          # Nasdaq-100
    "IWM": "IWM",          # Russell 2000 (small caps)
    "VIX": "^VIX",         # Volatility index
    "DXY": "UUP",          # US Dollar bullish ETF (proxy for DXY)
    "TLT": "TLT",          # 20+ year treasuries
    "IEF": "IEF",          # 7-10 year treasuries
    "TNX": "^TNX",         # 10-year treasury yield
    "USO": "USO",          # Oil
    "GLD": "GLD",          # Gold
    "SLV": "SLV",          # Silver
    "DBC": "DBC",          # Broad commodities
    "XLE": "XLE",          # Energy sector
    "XLF": "XLF",          # Financials sector
    "XLK": "XLK",          # Tech sector
    "XLU": "XLU",          # Utilities (defensive)
    "HYG": "HYG",          # High yield credit
}

ALL_TICKERS: list[str] = [TARGET_TICKER] + list(FEATURE_TICKERS.values())

# Default history window for training
DEFAULT_START = "2010-01-01"

# Directory layout
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "prices.parquet"


@dataclass
class ModelConfig:
    """Hyperparameters and settings for training."""

    lookback_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    ma_windows: tuple[int, ...] = (5, 10, 20, 50)
    vol_windows: tuple[int, ...] = (10, 20)
    target_horizon: int = 1  # predict 1 day ahead
    test_size: float = 0.2
    random_state: int = 42
    # Direction classification threshold: count a day as "up" only if
    # next-day return exceeds this (filters out noise). 0 = simple sign.
    direction_threshold: float = 0.0
