"""Market structure: swings, HH/HL/LH/LL, support/resistance, breakouts, sweeps."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

Pivots = List[Tuple[int, float]]  # (position in the passed frame, price)


def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[Pivots, Pivots]:
    """Confirmed swing highs/lows. A pivot needs `right` later candles to be confirmed,
    so the newest `right` candles can never be pivots (no look-ahead)."""
    n, w = len(df), left + right + 1
    if n < w:
        return [], []
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    hi_idx = np.where(sliding_window_view(hi, w).argmax(axis=1) == left)[0] + left
    lo_idx = np.where(sliding_window_view(lo, w).argmin(axis=1) == left)[0] + left
    return [(int(i), float(hi[i])) for i in hi_idx], [(int(i), float(lo[i])) for i in lo_idx]


def market_structure(highs: Pivots, lows: Pivots) -> Dict[str, str]:
    if len(highs) < 2 or len(lows) < 2:
        return {"trend": "unknown", "high_label": "n/a", "low_label": "n/a",
                "text": "Not enough swing points to judge structure"}
    hl = "HH" if highs[-1][1] > highs[-2][1] else "LH"
    ll = "HL" if lows[-1][1] > lows[-2][1] else "LL"
    if hl == "HH" and ll == "HL":
        trend = "uptrend"
    elif hl == "LH" and ll == "LL":
        trend = "downtrend"
    else:
        trend = "mixed"
    names = {"HH": "higher high", "LH": "lower high", "HL": "higher low", "LL": "lower low"}
    return {"trend": trend, "high_label": hl, "low_label": ll,
            "text": f"Last swing high is a {names[hl]}, last swing low is a {names[ll]}"}


def nearest_levels(price: float, highs: Pivots, lows: Pivots, n_recent: int = 6):
    """(nearest resistance above price, nearest support below price) from recent pivots."""
    res = [p for _, p in highs[-n_recent:] if p > price]
    sup = [p for _, p in lows[-n_recent:] if p < price]
    return (min(res) if res else None), (max(sup) if sup else None)


def detect_breakout_retest(df: pd.DataFrame, highs: Pivots, lows: Pivots,
                           lookback: int = 12, tol_atr: float = 0.25) -> Dict[str, Optional[dict]]:
    """Look for a close through a swing level in the last `lookback` candles, then a retest.
    state: 'breakout' (no retest yet), 'retest_confirmed', or 'failed' (closed back through)."""
    out: Dict[str, Optional[dict]] = {"long": None, "short": None}
    n = len(df)
    if n < 5:
        return out
    c, o = df["close"].to_numpy(float), df["open"].to_numpy(float)
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    rv = df["rel_vol"].to_numpy(float) if "rel_vol" in df else np.full(n, np.nan)
    tol = tol_atr * float(df["atr"].iloc[-1])
    start = max(1, n - lookback)
    for direction, pivots in (("long", highs[-4:]), ("short", lows[-4:])):
        long = direction == "long"
        found = None
        for pos, level in reversed(pivots):
            for b in range(max(start, pos + 1), n):
                broke = (c[b] > level and c[b - 1] <= level) if long else (c[b] < level and c[b - 1] >= level)
                if not broke:
                    continue
                after = slice(b + 1, n)
                held = bool(np.all(c[b:] > level)) if long else bool(np.all(c[b:] < level))
                touched = bool(np.any(lo[after] <= level + tol)) if long else bool(np.any(hi[after] >= level - tol))
                confirm = c[-1] > o[-1] if long else c[-1] < o[-1]
                if not held:
                    state = "failed"
                elif b < n - 1 and touched and confirm:
                    state = "retest_confirmed"
                else:
                    state = "breakout"
                found = {"level": level, "state": state, "bars_ago": n - 1 - b,
                         "volume_ok": bool(rv[b] >= 1.0) if not np.isnan(rv[b]) else False}
                break
            if found:
                break
        out[direction] = found
    return out


def liquidity_sweep(df: pd.DataFrame, highs: Pivots, lows: Pivots, lookback: int = 3) -> Dict[str, Optional[float]]:
    """Wick through a swing level that closes back inside it (stop-run).
    'bullish' = swing low swept then reclaimed; 'bearish' = swing high swept then rejected."""
    res: Dict[str, Optional[float]] = {"bullish": None, "bearish": None}
    n = len(df)
    hi, lo, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    for k in range(max(1, n - lookback), n):
        for pos, lvl in lows[-6:]:
            if pos < k and lo[k] < lvl < c[k]:
                res["bullish"] = lvl
        for pos, lvl in highs[-6:]:
            if pos < k and hi[k] > lvl > c[k]:
                res["bearish"] = lvl
    return res


def false_breakout(df: pd.DataFrame, highs: Pivots, lows: Pivots, lookback: int = 6) -> Dict[str, Optional[float]]:
    """Candle(s) CLOSED beyond a swing level recently, but price is now back inside.
    'up' = failed break above a high (bull trap); 'down' = failed break below a low (bear trap)."""
    res: Dict[str, Optional[float]] = {"up": None, "down": None}
    n = len(df)
    c = df["close"].to_numpy(float)
    rng = range(max(1, n - lookback), n - 1)
    for pos, lvl in highs[-4:]:
        if any(c[k] > lvl for k in rng if k > pos) and c[-1] < lvl:
            res["up"] = lvl
    for pos, lvl in lows[-4:]:
        if any(c[k] < lvl for k in rng if k > pos) and c[-1] > lvl:
            res["down"] = lvl
    return res
