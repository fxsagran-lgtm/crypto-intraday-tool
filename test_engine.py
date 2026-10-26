"""Tests for the signal engine and backtester (synthetic data is used ONLY in tests)."""
import unittest

import pandas as pd

from backtest import _stats, run_backtest
from config import Params
from indicators import add_indicators
from risk import position_size
from signals import TradeSignal, analyze
from synth import make_5m, to_15m


def flat_df(n=40):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10.0}, index=idx)


def one_shot_long():
    fired = {"done": False}
    def fn(a, b):
        if fired["done"]:
            return TradeSignal("T")
        fired["done"] = True
        return TradeSignal("T", action="LONG", entry=100.0, stop=98.0, targets=[102, 104, 106], score=7,
                           indicators={"5m": {"atr": 1.0}})
    return fn


P = Params(min_bars_5m=10, max_hold_bars=5)


def run(df):
    return run_backtest("T", df, to_15m(df), P, balance=1000, risk_pct=1.0, signal_fn=one_shot_long())


class BacktestMechanicsTests(unittest.TestCase):
    def test_stop_loses_exactly_planned_risk_including_costs(self):
        df = flat_df(); df.iloc[12, df.columns.get_loc("low")] = 97.5
        r = run(df)
        t = r.trades.iloc[0]
        self.assertEqual(t["exit_reason"], "stop")
        self.assertAlmostEqual(t["pnl"], -10.0, places=6)
        self.assertAlmostEqual(t["r_multiple"], -1.0, places=6)

    def test_target_pnl_matches_formula(self):
        df = flat_df(); df.iloc[12, df.columns.get_loc("high")] = 104.2
        t = run(df).trades.iloc[0]
        size = position_size(1000, 1.0, 100.0, 98.0, "long", P.fee_rate, P.slippage_rate, 5.0)
        fill = 100 * (1 + P.slippage_rate)
        expected = size["qty"] * (104 - fill) - P.fee_rate * size["qty"] * (fill + 104)
        self.assertEqual(t["exit_reason"], "target")
        self.assertAlmostEqual(t["pnl"], expected, places=6)
        self.assertGreater(t["pnl"], 0)

    def test_stop_wins_when_both_touched_in_one_candle(self):
        df = flat_df(); i = df.columns.get_loc
        df.iloc[11, i("low")], df.iloc[11, i("high")] = 97.0, 105.0
        self.assertEqual(run(df).trades.iloc[0]["exit_reason"], "stop")

    def test_gap_through_stop_skips_trade(self):
        df = flat_df(); df.iloc[11, df.columns.get_loc("open")] = 97.0
        self.assertEqual(len(run(df).trades), 0)

    def test_time_exit(self):
        t = run(flat_df()).trades.iloc[0]
        self.assertEqual(t["exit_reason"], "time")
        self.assertEqual(t["bars_held"], 5)

    def test_fills_use_next_candle_open_not_signal_close(self):
        df = flat_df(); df.iloc[11, df.columns.get_loc("open")] = 100.3
        t = run(df).trades.iloc[0]
        self.assertAlmostEqual(t["entry"], 100.3 * (1 + P.slippage_rate))
        self.assertEqual(t["entry_time"], df.index[11])

    def test_stats(self):
        trades = pd.DataFrame({"pnl": [20.0, -10.0, 20.0, -10.0], "r_multiple": [2, -1, 2, -1], "fees": [1.0] * 4})
        eq = pd.Series([1000, 1020, 1010, 1030, 1020.0])
        s = _stats(trades, eq, 1000.0, 100)
        self.assertEqual(s["total_trades"], 4)
        self.assertAlmostEqual(s["win_rate_pct"], 50.0)
        self.assertAlmostEqual(s["profit_factor"], 2.0)
        self.assertAlmostEqual(s["max_drawdown_pct"], (1020 - 1010) / 1020 * 100)
        self.assertAlmostEqual(s["net_return_pct"], 2.0)
        self.assertTrue(s["low_sample"])

    def test_no_trades_stats(self):
        s = _stats(pd.DataFrame(), pd.Series([1000.0]), 1000.0, 10)
        self.assertEqual(s["total_trades"], 0)
        self.assertIsNone(s["win_rate_pct"])


class SignalEngineTests(unittest.TestCase):
    def test_insufficient_data_is_no_trade(self):
        d5 = make_5m(60)
        s = analyze("X", d5, to_15m(d5))
        self.assertEqual(s.action, "NO TRADE")
        self.assertIn("Not enough data", s.no_trade_reasons[0])

    def test_signals_obey_invariants(self):
        p, found = Params(), 0
        for seed in (1, 2):
            raw = make_5m(4000, seed=seed)
            d5, d15 = add_indicators(raw, p), add_indicators(to_15m(raw), p)
            close15 = d15.index + pd.Timedelta(minutes=15)
            for i in range(200, 4000, 2):
                n15 = int(close15.searchsorted(d5.index[i] + pd.Timedelta(minutes=5), side="right"))
                s = analyze("X", d5.iloc[max(0, i + 1 - 300): i + 1], d15.iloc[max(0, n15 - 200): n15], p, prepared=True)
                if s.action == "NO TRADE":
                    self.assertTrue(s.no_trade_reasons)
                    self.assertEqual(s.targets, [])
                    continue
                found += 1
                atr = s.indicators["5m"]["atr"]
                sg = 1 if s.action == "LONG" else -1
                risk = sg * (s.entry - s.stop)
                self.assertGreater(risk, 0)
                self.assertTrue(p.min_risk_atr * atr <= risk <= p.max_risk_atr * atr + 1e-9)
                self.assertTrue(sg * (s.targets[0] - s.entry) > 0)
                self.assertTrue(s.targets[0] * sg < s.targets[1] * sg < s.targets[2] * sg)
                self.assertGreaterEqual(s.score, p.min_score)
                self.assertGreaterEqual(s.rr_net_t2, p.min_net_rr)
                self.assertTrue(s.reasons and s.conditions and s.invalidation)
        self.assertGreater(found, 0, "synthetic scan produced no signals to check")

    def test_signal_ignores_future_candles(self):
        p, raw = Params(), make_5m(2500, seed=3)
        d15raw = to_15m(raw)
        for i in (800, 1500, 2200):
            cut = raw.iloc[: i + 1]
            end = cut.index[-1] + pd.Timedelta(minutes=5)
            c15 = d15raw[d15raw.index + pd.Timedelta(minutes=15) <= end]
            a = analyze("X", cut, c15, p)                                  # sees only the past
            full5, full15 = add_indicators(raw, p), add_indicators(d15raw, p)
            n15 = int((full15.index + pd.Timedelta(minutes=15)).searchsorted(end, side="right"))
            b = analyze("X", full5.iloc[: i + 1], full15.iloc[:n15], p, prepared=True)   # sliced from the future-containing frame
            self.assertEqual((a.action, a.entry, a.stop, a.score), (b.action, b.entry, b.stop, b.score))


class NoLookAheadBacktestTests(unittest.TestCase):
    def test_truncated_run_matches_prefix_of_full_run(self):
        raw = make_5m(3600, seed=1)
        full = run_backtest("X", raw, to_15m(raw), Params())
        cut = raw.iloc[:2600]
        part = run_backtest("X", cut, to_15m(cut), Params())
        end = cut.index[-1] + pd.Timedelta(minutes=5)
        a = part.trades[(part.trades["exit_reason"] != "end_of_data")] if len(part.trades) else part.trades
        b = full.trades[full.trades["exit_time"] <= end] if len(full.trades) else full.trades
        self.assertGreater(len(a), 0, "need at least one trade to compare")
        cols = ["entry_time", "exit_time", "side", "entry", "exit", "pnl"]
        pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
