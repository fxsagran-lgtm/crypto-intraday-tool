"""Backtest on REAL downloaded candles (run download_history.py first). Includes fees and slippage."""
import argparse
import json
import os

import pandas as pd

from backtest import run_backtest
from config import Params, SYMBOLS


def load(sym: str, res: str):
    path = os.path.join("data", "real", f"{sym}_{res}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run:  python download_history.py --days 14")
    with open(path + ".meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    if not meta.get("real_market_data"):
        raise SystemExit(f"{path} is not marked as real market data; refusing to use it.")
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    return df, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSD", choices=list(SYMBOLS.values()))
    ap.add_argument("--balance", type=float, default=1000.0)
    ap.add_argument("--risk", type=float, default=1.0, help="risk per trade, percent of balance")
    ap.add_argument("--fee", type=float, default=Params().fee_rate, help="fee per side, e.g. 0.0005 = 0.05%%")
    ap.add_argument("--slippage", type=float, default=Params().slippage_rate)
    ap.add_argument("--target-r", type=float, default=2.0, choices=[1.0, 2.0, 3.0])
    args = ap.parse_args()
    d5, m5 = load(args.symbol, "5m")
    d15, _ = load(args.symbol, "15m")
    p = Params(fee_rate=args.fee, slippage_rate=args.slippage, target_r=args.target_r)
    res = run_backtest(args.symbol, d5, d15, p, args.balance, args.risk)
    s = res.stats
    print(res.label)
    print(f"Data: {m5['source']} | {m5['first_candle']} -> {m5['last_candle']} | missing candles: {m5['missing_candles']}")
    print(f"Fee/side {p.fee_rate:.4%}, slippage {p.slippage_rate:.4%}, risk {args.risk}% per trade, target {p.target_r}R")
    pf = "n/a (no losing trades)" if s["profit_factor"] is None and s["total_trades"] else s["profit_factor"]
    print(f"Trades {s['total_trades']} | win rate {s['win_rate_pct']} | profit factor {pf}")
    print(f"Net return {s['net_return_pct']:.2f}% | max drawdown {s['max_drawdown_pct']:.2f}% | fees paid {s['fees_paid']:.2f}")
    if s["low_sample"]:
        print("WARNING: fewer than 30 trades. These statistics are NOT reliable evidence of anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
