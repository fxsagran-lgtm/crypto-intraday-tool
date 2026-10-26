"""Market-condition classifier: trending / ranging / volatility / volume."""
import numpy as np
import pandas as pd

from config import Params


def classify_regime(df: pd.DataFrame, p: Params = Params()) -> dict:
    """`df` must already contain indicator columns (see indicators.add_indicators)."""
    last = df.iloc[-1]
    needed = [last.get("er"), last.get("atr_pct")]
    if any(v is None or pd.isna(v) for v in needed):
        return {"label": "Unknown", "trending": False, "ranging": False, "high_volatility": False,
                "low_volume": False, "er": None, "atr_ratio": None, "rel_vol_5": None,
                "tradeable": False, "block_reasons": ["Not enough data to classify the market"]}
    er = float(last["er"])
    base = df["atr_pct"].tail(100).median()
    atr_ratio = float(last["atr_pct"] / base) if base and not np.isnan(base) else 1.0
    rel_vol_5 = float(df["rel_vol"].tail(5).mean())
    trending, ranging = er >= p.er_trend, er <= p.er_range
    high_vol = atr_ratio >= p.vol_high_ratio
    low_vol = (not np.isnan(rel_vol_5)) and rel_vol_5 < p.low_vol_rel
    label = "Trending" if trending else "Ranging" if ranging else "Transitional"
    blocks = []
    if ranging:
        blocks.append(f"Market is ranging (efficiency ratio {er:.2f} <= {p.er_range})")
    if high_vol:
        blocks.append(f"Volatility is very high ({atr_ratio:.1f}x its recent norm)")
    if low_vol:
        blocks.append(f"Volume is thin ({rel_vol_5:.2f}x its 20-candle average)")
    return {"label": label, "trending": trending, "ranging": ranging, "high_volatility": high_vol,
            "low_volume": low_vol, "er": er, "atr_ratio": atr_ratio, "rel_vol_5": rel_vol_5,
            "tradeable": not blocks, "block_reasons": blocks}
