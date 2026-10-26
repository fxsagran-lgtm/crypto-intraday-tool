"""Paper trading only: simulated positions stored in a local JSON file. No exchange connection exists."""
import json
import os
from typing import List

import pandas as pd

from config import Params
from risk import position_size
from signals import TradeSignal

TF5 = pd.Timedelta(minutes=5)


class PaperTrader:
    def __init__(self, path: str = "paper_trades.json", start_balance: float = 1000.0):
        self.path = path
        self.state = {"start_balance": start_balance, "balance": start_balance, "open": [], "closed": []}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                self.state = json.load(fh)

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2)
        os.replace(tmp, self.path)

    def has_open(self, symbol: str) -> bool:
        return any(t["symbol"] == symbol for t in self.state["open"])

    def open_from_signal(self, sig: TradeSignal, risk_pct: float, p: Params = Params(),
                         max_leverage: float = 5.0) -> dict:
        if sig.action == "NO TRADE":
            raise ValueError("Cannot paper-trade a NO TRADE signal.")
        if self.has_open(sig.symbol):
            raise ValueError(f"A paper position in {sig.symbol} is already open.")
        side = sig.action.lower()
        sgn = 1 if side == "long" else -1
        size = position_size(self.state["balance"], risk_pct, sig.entry, sig.stop, side,
                             p.fee_rate, p.slippage_rate, max_leverage)
        target = sig.entry + sgn * p.target_r * (sgn * (sig.entry - sig.stop))
        trade = {"symbol": sig.symbol, "side": side, "entry": sig.entry * (1 + p.slippage_rate * sgn),
                 "stop": sig.stop, "target": target, "qty": size["qty"],
                 "risk_amount": size["risk_amount_at_stop"], "score": sig.score,
                 "signal_candle": str(sig.candle_time),
                 "start_after": str(sig.candle_time + TF5), "max_hold_bars": p.max_hold_bars}
        self.state["open"].append(trade)
        self.save()
        return trade

    def update(self, symbol: str, df5: pd.DataFrame, p: Params = Params()) -> List[dict]:
        """Walk closed 5m candles after each open paper trade's signal; stop wins ties."""
        closed_now, keep = [], []
        for t in self.state["open"]:
            if t["symbol"] != symbol:
                keep.append(t)
                continue
            sgn = 1 if t["side"] == "long" else -1
            bars = df5[df5.index >= pd.Timestamp(t["start_after"])]
            result = None
            for n, (ts, row) in enumerate(bars.iterrows(), start=1):
                stop_hit = row["low"] <= t["stop"] if sgn == 1 else row["high"] >= t["stop"]
                tgt_hit = row["high"] >= t["target"] if sgn == 1 else row["low"] <= t["target"]
                if stop_hit:
                    result = (t["stop"] * (1 - p.slippage_rate * sgn), "stop", ts)
                elif tgt_hit:
                    result = (t["target"], "target", ts)
                elif n >= t["max_hold_bars"]:
                    result = (row["close"] * (1 - p.slippage_rate * sgn), "time", ts)
                if result:
                    break
            if not result:
                keep.append(t)
                continue
            px, why, ts = result
            fees = p.fee_rate * t["qty"] * (t["entry"] + px)
            pnl = sgn * t["qty"] * (px - t["entry"]) - fees
            done = {**t, "exit": float(px), "exit_reason": why, "exit_time": str(ts + TF5),
                    "fees": fees, "pnl": pnl, "r_multiple": pnl / t["risk_amount"]}
            self.state["balance"] += pnl
            self.state["closed"].append(done)
            closed_now.append(done)
        self.state["open"] = keep
        if closed_now:
            self.save()
        return closed_now
