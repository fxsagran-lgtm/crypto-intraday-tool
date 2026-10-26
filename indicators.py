"""Indicators implemented directly in pandas/numpy (causal: no future data is used)."""
import numpy as np
import pandas as pd

from config import Params

OHLCV = ["open", "high", "low", "close", "volume"]


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder Average True Range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored to each UTC day (resets at 00:00 UTC)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    day = df.index.normalize()
    cum_pv = (typical * df["volume"]).groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def efficiency_ratio(close: pd.Series, length: int = 20) -> pd.Series:
    """Kaufman efficiency ratio: ~1 = straight-line trend, ~0 = choppy/ranging."""
    net = (close - close.shift(length)).abs()
    path = close.diff().abs().rolling(length, min_periods=length).sum()
    return net / path.replace(0, np.nan)


def add_indicators(df: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex (candle open time, UTC).")
    out = df.copy()
    out["ema9"] = ema(out["close"], 9)
    out["ema21"] = ema(out["close"], 21)
    out["ema50"] = ema(out["close"], 50)
    out["rsi"] = rsi(out["close"], p.rsi_len)
    out["atr"] = atr(out, p.atr_len)
    out["atr_pct"] = out["atr"] / out["close"]
    out["vwap"] = vwap(out)
    out["vol_ma"] = out["volume"].rolling(p.vol_len, min_periods=p.vol_len).mean()
    out["rel_vol"] = out["volume"] / out["vol_ma"].replace(0, np.nan)
    out["er"] = efficiency_ratio(out["close"], p.er_len)
    return out
