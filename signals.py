"""Signal engine: 15m sets the bias, 5m supplies the entry, a score decides, NO TRADE is the default."""
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from config import Params
from indicators import add_indicators
from regime import classify_regime
from risk import net_reward_risk
from structure import (detect_breakout_retest, false_breakout, liquidity_sweep,
                       market_structure, swing_points)

MAX_SCORE = 11
IND_KEYS = ["close", "ema9", "ema21", "ema50", "rsi", "atr", "vwap", "rel_vol", "er"]


@dataclass
class TradeSignal:
    symbol: str
    action: str = "NO TRADE"                 # LONG / SHORT / NO TRADE
    candle_time: Optional[pd.Timestamp] = None   # open time of the last CLOSED 5m candle used
    entry: Optional[float] = None
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    stop: Optional[float] = None
    targets: List[float] = field(default_factory=list)
    rr_gross: List[float] = field(default_factory=list)
    rr_net_t2: Optional[float] = None
    invalidation: str = ""
    score: int = 0
    max_score: int = MAX_SCORE
    reasons: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    no_trade_reasons: List[str] = field(default_factory=list)
    bias: str = "neutral"
    bias_score: int = 0
    bias_parts: List[str] = field(default_factory=list)
    timeframe_agreement: str = ""
    structure_5m: str = ""
    structure_15m: str = ""
    regime_5m: dict = field(default_factory=dict)
    regime_15m: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    indicators: dict = field(default_factory=dict)


def _bias(ms15: dict, l15) -> tuple:
    score, parts = 0, []
    if ms15["trend"] == "uptrend":
        score += 2; parts.append("15m structure makes higher highs and higher lows (+2)")
    elif ms15["trend"] == "downtrend":
        score -= 2; parts.append("15m structure makes lower highs and lower lows (-2)")
    if l15["ema9"] > l15["ema21"] > l15["ema50"]:
        score += 1; parts.append("15m EMAs stacked bullish 9 > 21 > 50 (+1)")
    elif l15["ema9"] < l15["ema21"] < l15["ema50"]:
        score -= 1; parts.append("15m EMAs stacked bearish 9 < 21 < 50 (-1)")
    if l15["close"] > l15["vwap"]:
        score += 1; parts.append("15m close above VWAP (+1)")
    elif l15["close"] < l15["vwap"]:
        score -= 1; parts.append("15m close below VWAP (-1)")
    if l15["rsi"] > 50:
        score += 1; parts.append("15m RSI above 50 (+1)")
    elif l15["rsi"] < 50:
        score -= 1; parts.append("15m RSI below 50 (-1)")
    return score, parts


def analyze(symbol: str, df5: pd.DataFrame, df15: pd.DataFrame, p: Params = Params(),
            prepared: bool = False) -> TradeSignal:
    """df5/df15 must contain only CLOSED candles. Everything here uses data up to the last row only."""
    sig = TradeSignal(symbol=symbol)
    if not prepared:
        df5, df15 = add_indicators(df5, p), add_indicators(df15, p)
    if len(df5):
        sig.candle_time = df5.index[-1]
    if len(df5) < p.min_bars_5m or len(df15) < p.min_bars_15m:
        sig.no_trade_reasons.append(
            f"Not enough data: need {p.min_bars_5m} x 5m and {p.min_bars_15m} x 15m candles, "
            f"have {len(df5)} and {len(df15)}.")
        return sig
    l5, l15 = df5.iloc[-1], df15.iloc[-1]
    if l5[IND_KEYS].isna().any() or l15[IND_KEYS].isna().any():
        sig.no_trade_reasons.append("Indicators are still warming up (missing values in latest candle).")
        return sig

    # ---------- context ----------
    w5, w15 = df5.tail(p.structure_lookback), df15.tail(p.structure_lookback)
    h5, lw5 = swing_points(w5, p.swing_left, p.swing_right)
    h15, lw15 = swing_points(w15, p.swing_left, p.swing_right)
    ms5, ms15 = market_structure(h5, lw5), market_structure(h15, lw15)
    sig.structure_5m, sig.structure_15m = ms5["trend"], ms15["trend"]
    sig.bias_score, sig.bias_parts = _bias(ms15, l15)
    sig.bias = ("long" if sig.bias_score >= p.bias_threshold
                else "short" if sig.bias_score <= -p.bias_threshold else "neutral")
    sig.regime_15m, sig.regime_5m = classify_regime(df15, p), classify_regime(df5, p)
    sweep, fb = liquidity_sweep(w5, h5, lw5), false_breakout(w5, h5, lw5)
    breakout = detect_breakout_retest(w5, h5, lw5)
    sig.flags = {"sweep": sweep, "false_breakout": fb, "breakout": breakout,
                 "structure_text_5m": ms5["text"], "structure_text_15m": ms15["text"]}
    sig.indicators = {"5m": {k: float(l5[k]) for k in IND_KEYS}, "15m": {k: float(l15[k]) for k in IND_KEYS}}

    # ---------- gates: any failure => NO TRADE ----------
    r = sig.no_trade_reasons
    if not sig.regime_15m["tradeable"]:
        r += [f"15m: {x}" for x in sig.regime_15m["block_reasons"]]
    if sig.regime_5m["low_volume"] or sig.regime_5m["high_volatility"]:
        r += [f"5m: {x}" for x in sig.regime_5m["block_reasons"] if "ranging" not in x]
    if sig.bias == "neutral":
        sig.timeframe_agreement = "15m has no clear direction, so there is nothing for the 5m chart to align with."
        r.append(f"15m bias is unclear (score {sig.bias_score:+d}, need +/-{p.bias_threshold}).")
    else:
        opposite = "downtrend" if sig.bias == "long" else "uptrend"
        if ms5["trend"] == opposite:
            sig.timeframe_agreement = (f"CONFLICT: 15m bias is {sig.bias.upper()} but 5m structure is a {ms5['trend']}.")
            r.append(sig.timeframe_agreement)
        else:
            sig.timeframe_agreement = (f"AGREE: 15m bias is {sig.bias.upper()} and 5m structure is not against it "
                                       f"({ms5['trend']}).")
    if r:
        return sig

    # ---------- 5m trigger ----------
    is_long = sig.bias == "long"
    sgn = 1 if is_long else -1
    atr5, close = float(l5["atr"]), float(l5["close"])
    rec = w5.tail(4)
    if is_long:
        touched = bool((rec["low"] <= rec["ema21"] + 0.3 * atr5).any())
        trig_a = bool(l5["ema21"] > l5["ema50"] and touched and close > l5["ema9"]
                      and close > l5["ema21"] and close > l5["open"])
    else:
        touched = bool((rec["high"] >= rec["ema21"] - 0.3 * atr5).any())
        trig_a = bool(l5["ema21"] < l5["ema50"] and touched and close < l5["ema9"]
                      and close < l5["ema21"] and close < l5["open"])
    ev = breakout["long" if is_long else "short"]
    trig_b = bool(ev and ev["state"] == "retest_confirmed")
    if not (trig_a or trig_b):
        r.append("No 5m trigger: no EMA-21 pullback bounce and no confirmed breakout-retest.")
        return sig
    if abs(close - float(l5["ema21"])) > p.max_extension_atr * atr5:
        r.append(f"Price is more than {p.max_extension_atr} ATR away from the 5m EMA 21 (chasing risk).")
        return sig
    trap = fb["up"] if is_long else fb["down"]
    if trap is not None:
        r.append(f"Possible false breakout / trap around {trap:,.2f} in the last 6 candles.")
        return sig

    # ---------- score ----------
    score = 2
    sig.reasons.append(f"15m bias is {sig.bias.upper()} (score {sig.bias_score:+d}); 5m agrees (+2)")
    if abs(sig.bias_score) >= 4:
        score += 1; sig.reasons.append("15m bias is strong (+1)")
    if trig_a:
        score += 2; sig.reasons.append("5m pullback to EMA 21 and bounce candle closed back above EMA 9 (+2)"
                                       if is_long else
                                       "5m pullback to EMA 21 and rejection candle closed below EMA 9 (+2)")
    if trig_b:
        score += 2; sig.reasons.append(f"Breakout of {ev['level']:,.2f} retested and holding (+2)")
    sw = sweep["bullish" if is_long else "bearish"]
    if sw is not None:
        score += 1; sig.reasons.append(f"Liquidity sweep around {sw:,.2f} was reclaimed/rejected (+1)")
    if l5["rel_vol"] >= 1.1:
        score += 1; sig.reasons.append(f"Volume confirms: {l5['rel_vol']:.2f}x average (+1)")
    if (close > l5["vwap"]) == is_long and close != l5["vwap"]:
        score += 1; sig.reasons.append(f"5m price on the {'long' if is_long else 'short'} side of VWAP (+1)")
    lo_r, hi_r = (40, 68) if is_long else (32, 60)
    if lo_r <= l5["rsi"] <= hi_r:
        score += 1; sig.reasons.append(f"5m RSI {l5['rsi']:.0f} is in a healthy zone, not stretched (+1)")
    sig.score = score
    if score < p.min_score:
        r.append(f"Setup score {score}/{MAX_SCORE} is below the minimum of {p.min_score}.")
        return sig

    # ---------- levels ----------
    entry = close
    away = entry - sgn * 0.3 * atr5
    sig.entry_low, sig.entry_high = (away, entry) if is_long else (entry, away)
    n5 = len(w5)
    if is_long:   # pullback/sweep low of the last 8 candles, plus a confirmed swing low only if it is recent
        recent = [x for pos, x in lw5 if pos >= n5 - 20 and x < entry]
        struct = min([float(w5["low"].tail(8).min())] + recent[-1:])
    else:
        recent = [x for pos, x in h5 if pos >= n5 - 20 and x > entry]
        struct = max([float(w5["high"].tail(8).max())] + recent[-1:])
    stop = struct - sgn * p.atr_stop_buffer * atr5
    risk = sgn * (entry - stop)
    if not (p.min_risk_atr * atr5 <= risk <= p.max_risk_atr * atr5):
        r.append(f"Stop distance {risk / atr5:.2f} ATR is outside the allowed "
                 f"{p.min_risk_atr}-{p.max_risk_atr} ATR range.")
        return sig
    # room check uses 15m swing levels only: tiny 5m swings next to price are noise, not real resistance
    opp = ([x for _, x in h15 if x > entry] if is_long else [x for _, x in lw15 if x < entry])
    if opp:
        lvl = min(opp) if is_long else max(opp)
        room = sgn * (lvl - entry) / risk
        if room < p.min_room_r:
            r.append(f"Nearest {'resistance' if is_long else 'support'} at {lvl:,.2f} leaves only {room:.1f}R of room.")
            return sig
        sig.reasons.append(f"Nearest {'resistance' if is_long else 'support'} {lvl:,.2f} is {room:.1f}R away (room to run)")
    targets = [entry + sgn * k * risk for k in (1, 2, 3)]
    side = "long" if is_long else "short"
    sig.rr_net_t2 = net_reward_risk(entry, stop, targets[1], side, p.fee_rate, p.slippage_rate)
    if not (sig.rr_net_t2 >= p.min_net_rr):
        r.append(f"After fees and slippage, reward:risk to Target 2 is only {sig.rr_net_t2:.2f}.")
        return sig

    sig.action, sig.entry, sig.stop, sig.targets = side.upper(), entry, stop, targets
    sig.rr_gross = [1.0, 2.0, 3.0]
    sig.invalidation = (f"A 5m candle CLOSES {'below' if is_long else 'above'} {stop:,.2f}, "
                        f"or the 15m bias score drops to {'0 or below' if is_long else '0 or above'}.")
    sig.conditions = [
        "Wait for the current 5m candle to CLOSE before acting.",
        f"Enter only inside {sig.entry_low:,.2f} - {sig.entry_high:,.2f}.",
        f"Skip it if price runs more than {p.max_chase_atr} ATR ({p.max_chase_atr * atr5:,.2f}) beyond the zone (do not chase).",
        f"15m bias must still be {sig.bias.upper()} when you enter.",
        "Volume should not dry up right before entry.",
    ]
    return sig
