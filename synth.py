"""Synthetic candle generator FOR TESTS ONLY. Never used by the live dashboard."""
import numpy as np
import pandas as pd


def make_5m(n=6000, seed=1, start="2026-01-01", base=60000.0):
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    i = 0
    while i < n:                      # regimes: up / down / flat, 100-400 candles each
        length = int(rng.integers(100, 400))
        drift[i:i + length] = rng.choice([0.00035, -0.00035, 0.0])
        i += length
    vol = 0.0012 * (1 + 0.5 * np.sin(np.arange(n) / 300))
    ret = drift + rng.normal(0, vol)
    close = base * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[base], close[:-1]])
    wick = np.abs(rng.normal(0, vol * 0.6, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - np.abs(rng.normal(0, vol * 0.6, n)) * close
    volume = rng.lognormal(3, 0.4, n) * (1 + 40 * np.abs(ret) / vol.mean())
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC", name="time")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def to_15m(df5):
    g = df5.resample("15min", label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(), "volume": g["volume"].sum()})
    return out.dropna()
