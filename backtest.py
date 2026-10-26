"""Candle-by-candle backtester. Decisions at candle i use only data up to the CLOSE of candle i;
fills happen at the OPEN of candle i+1. Backtest results are simulations, never live performance."""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import Params
from indicators import add_indicators
from risk import position_size
from signals import TradeSignal, analyze

LABEL = "BACKTEST: historical simulation on past candles. Not live results, and no guide to future performance."


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict
    label: str = LABEL


def _stats(trades: pd.DataFrame, equity: pd.Series, start_balance: float, bars: int) -> dict:
    n = len(trades)
    out = {"total_trades": n, "bars_tested": bars, "start_balance": start_balance}
    if n == 0:
        out.update(win_rate_pct=None, profit_factor=None, max_drawdown_pct=0.0, net_return_pct=0.0,
                   net_pnl=0.0, avg_r=None, fees_paid=0.0, low_sample=True)
        return out
    wins = trades[trades["pnl"] > 0]["pnl"].sum()
    losses = -trades[trades["pnl"] < 0]["pnl"].sum()
    peak = equity.cummax()
    out.update(
        win_rate_pct=float((trades["pnl"] > 0).mean() * 100),
        profit_factor=float(wins / losses) if losses > 0 else None,   # None = no losing trades
        max_drawdown_pct=float(((peak - equity) / peak).max() * 100),
        net_return_pct=float((equity.iloc[-1] / start_balance - 1) * 100),
        net_pnl=float(equity.iloc[-1] - start_balance),
        avg_r=float(trades["r_multiple"].mean()),
        fees_paid=float(trades["fees"].sum()),
        low_sample=n < 30,
    )
    return out


def run_backtest(symbol: str, df5: pd.DataFrame, df15: pd.DataFrame, p: Params = Params(),
                 balance: float = 1000.0, risk_pct: float = 1.0, max_leverage: float = 5.0,
                 signal_fn: Optional[Callable[[pd.DataFrame, pd.DataFrame], TradeSignal]] = None) -> BacktestResult:
    d5, d15 = add_indicators(df5, p), add_indicators(df15, p)   # all indicators are causal
    fn = signal_fn or (lambda a, b: analyze(symbol, a, b, p, prepared=True))
    n = len(d5)
    o, h, l, c = (d5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    tf5, tf15 = pd.Timedelta(minutes=5), pd.Timedelta(minutes=15)
    close15 = d15.index + tf15          # moment each 15m candle closes
    equity, rows, times = balance, [], [d5.index[0]]
    curve = [balance]
    i = p.min_bars_5m
    while i < n - 1:
        known_until = d5.index[i] + tf5                          # moment candle i closes
        n15 = int(close15.searchsorted(known_until, side="right"))  # only fully closed 15m candles
        sig = fn(d5.iloc[max(0, i + 1 - 300): i + 1], d15.iloc[max(0, n15 - 200): n15])
        if sig.action == "NO TRADE":
            i += 1
            continue
        side = "long" if sig.action == "LONG" else "short"
        sgn = 1 if side == "long" else -1
        stop = sig.stop
        risk_dist = sgn * (sig.entry - stop)
        target = sig.entry + sgn * p.target_r * risk_dist
        atr5 = sig.indicators["5m"]["atr"]
        j = i + 1
        # skip if the next open already gapped through stop/target or ran away from the zone
        if (sgn * (o[j] - stop) <= 0 or sgn * (target - o[j]) <= 0
                or sgn * (o[j] - sig.entry) > p.max_chase_atr * atr5):
            i += 1
            continue
        try:
            size = position_size(equity, risk_pct, o[j], stop, side, p.fee_rate, p.slippage_rate, max_leverage)
        except ValueError:
            i += 1
            continue
        qty = size["qty"]
        fill = o[j] * (1 + p.slippage_rate * sgn)
        last = min(n, j + p.max_hold_bars) - 1
        exit_idx, exit_px = last, None
        reason = "time" if last - j + 1 >= p.max_hold_bars else "end_of_data"
        for k in range(j, last + 1):
            stop_hit = l[k] <= stop if sgn == 1 else h[k] >= stop
            target_hit = h[k] >= target if sgn == 1 else l[k] <= target
            if stop_hit:   # if both are touched in one candle, assume the stop came first
                exit_idx, exit_px, reason = k, stop * (1 - p.slippage_rate * sgn), "stop"
                break
            if target_hit:
                exit_idx, exit_px, reason = k, target, "target"
                break
        if exit_px is None:
            exit_px = c[exit_idx] * (1 - p.slippage_rate * sgn)
        fees = p.fee_rate * qty * (fill + exit_px)
        pnl = sgn * qty * (exit_px - fill) - fees
        equity += pnl
        rows.append({"entry_time": d5.index[j], "exit_time": d5.index[exit_idx] + tf5, "side": side,
                     "entry": fill, "exit": exit_px, "stop": stop, "target": target, "qty": qty,
                     "fees": fees, "pnl": pnl, "r_multiple": pnl / size["risk_amount_at_stop"],
                     "exit_reason": reason, "bars_held": exit_idx - j + 1, "score": sig.score})
        times.append(d5.index[exit_idx] + tf5)
        curve.append(equity)
        i = exit_idx
    trades = pd.DataFrame(rows)
    eq = pd.Series(curve, index=pd.DatetimeIndex(times), name="equity")
    return BacktestResult(trades, eq, _stats(trades, eq, balance, n))
