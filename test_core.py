"""Tests for indicators, risk maths, structure detection and regime classification."""
import unittest

import numpy as np
import pandas as pd

from config import Params
from indicators import add_indicators, atr, efficiency_ratio, ema, rsi, vwap
from regime import classify_regime
from risk import loss_per_unit_at_stop, net_reward_risk, position_size
from structure import (detect_breakout_retest, false_breakout, liquidity_sweep,
                       market_structure, swing_points)
from synth import make_5m


def frame(highs, lows, closes=None, opens=None, start="2026-01-01"):
    closes = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    opens = opens if opens is not None else closes
    idx = pd.date_range(start, periods=len(highs), freq="5min", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": np.ones(len(highs))}, index=idx)


class IndicatorTests(unittest.TestCase):
    def test_ema_matches_manual_recursion(self):
        x = pd.Series([1.0, 2, 3, 2, 5, 4, 6])
        a = 2 / (3 + 1)
        manual = [x[0]]
        for v in x[1:]:
            manual.append(a * v + (1 - a) * manual[-1])
        np.testing.assert_allclose(ema(x, 3).to_numpy(), manual)

    def test_rsi_extremes(self):
        up = pd.Series(np.arange(1.0, 60))
        self.assertAlmostEqual(rsi(up).iloc[-1], 100.0)
        self.assertAlmostEqual(rsi(-up + 100).iloc[-1], 0.0, places=6)
        self.assertAlmostEqual(rsi(pd.Series(np.ones(60))).iloc[-1], 50.0)

    def test_rsi_bounded_on_random_data(self):
        r = rsi(make_5m(500)["close"]).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_atr_of_constant_range(self):
        df = frame([11.0] * 40, [9.0] * 40, closes=[10.0] * 40)
        self.assertAlmostEqual(atr(df, 14).iloc[-1], 2.0)

    def test_vwap_matches_manual_and_resets_daily(self):
        df = frame([10, 12, 14, 16], [8, 10, 12, 14], closes=[9, 11, 13, 15])
        df["volume"] = [1.0, 2.0, 3.0, 4.0]
        typ = (df["high"] + df["low"] + df["close"]) / 3
        manual = (typ * df["volume"]).cumsum() / df["volume"].cumsum()
        np.testing.assert_allclose(vwap(df).to_numpy(), manual.to_numpy())
        # two days: second day's first VWAP equals that candle's typical price
        idx = pd.DatetimeIndex(["2026-01-01 23:55", "2026-01-02 00:00"], tz="UTC")
        d2 = pd.DataFrame({"open": [1, 5.0], "high": [2, 6.0], "low": [0, 4.0],
                           "close": [1, 5.0], "volume": [1.0, 1.0]}, index=idx)
        self.assertAlmostEqual(vwap(d2).iloc[1], 5.0)

    def test_efficiency_ratio(self):
        self.assertAlmostEqual(efficiency_ratio(pd.Series(np.arange(50.0)), 20).iloc[-1], 1.0)
        zig = pd.Series([1.0, 2.0] * 30)
        self.assertLess(efficiency_ratio(zig, 20).iloc[-1], 0.1)

    def test_indicators_are_causal(self):
        full = add_indicators(make_5m(600))
        part = add_indicators(make_5m(600).iloc[:400])
        cols = ["ema9", "ema21", "ema50", "rsi", "atr", "vwap", "rel_vol", "er"]
        pd.testing.assert_frame_equal(full.iloc[:400][cols], part[cols])

    def test_add_indicators_validates_input(self):
        with self.assertRaises(ValueError):
            add_indicators(pd.DataFrame({"close": [1, 2]}))


class RiskTests(unittest.TestCase):
    def test_long_loss_at_stop_equals_planned_risk(self):
        s = position_size(1000, 1.0, 100.0, 98.0, "long", 0.0005, 0.0003)
        self.assertAlmostEqual(s["risk_amount_at_stop"], 10.0)
        self.assertAlmostEqual(s["qty"] * loss_per_unit_at_stop(100, 98, "long", 0.0005, 0.0003), 10.0)

    def test_short_works_and_costs_shrink_size(self):
        no_cost = position_size(1000, 1.0, 100.0, 102.0, "short", 0, 0)["qty"]
        with_cost = position_size(1000, 1.0, 100.0, 102.0, "short", 0.001, 0.001)["qty"]
        self.assertAlmostEqual(no_cost, 5.0)
        self.assertLess(with_cost, no_cost)

    def test_leverage_cap(self):
        s = position_size(1000, 5.0, 100.0, 99.9, "long", 0, 0, max_leverage=3)
        self.assertTrue(s["capped_by_leverage"])
        self.assertAlmostEqual(s["notional"], 3000.0)
        self.assertLess(s["risk_amount_at_stop"], s["risk_amount_planned"])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            position_size(1000, 1, 100, 101, "long", 0, 0)      # stop above long entry
        with self.assertRaises(ValueError):
            position_size(-5, 1, 100, 99, "long", 0, 0)
        with self.assertRaises(ValueError):
            position_size(1000, 0, 100, 99, "long", 0, 0)
        with self.assertRaises(ValueError):
            position_size(1000, 1, 100, 99, "sideways", 0, 0)

    def test_net_rr_below_gross(self):
        self.assertAlmostEqual(net_reward_risk(100, 98, 104, "long", 0, 0), 2.0)
        self.assertLess(net_reward_risk(100, 98, 104, "long", 0.0005, 0.0003), 2.0)
        self.assertLess(net_reward_risk(100, 102, 96, "short", 0.0005, 0.0003), 2.0)


class StructureTests(unittest.TestCase):
    def test_swing_points_known_zigzag(self):
        highs = [1, 2, 3, 9, 3, 2, 1, 2, 3, 12, 3, 2, 1, 2, 3]
        lows = [h - 1 for h in highs]
        h, _ = swing_points(frame(highs, lows), 3, 3)
        self.assertEqual([p for p, _ in h], [3, 9])
        self.assertEqual([v for _, v in h], [9, 12])

    def test_newest_candles_never_pivots(self):
        df = make_5m(300)
        h, l = swing_points(df, 3, 3)
        self.assertTrue(all(p <= len(df) - 1 - 3 for p, _ in h + l))

    def test_market_structure_labels(self):
        self.assertEqual(market_structure([(1, 10), (5, 12)], [(2, 5), (6, 7)])["trend"], "uptrend")
        self.assertEqual(market_structure([(1, 12), (5, 10)], [(2, 7), (6, 5)])["trend"], "downtrend")
        self.assertEqual(market_structure([(1, 12), (5, 13)], [(2, 7), (6, 5)])["trend"], "mixed")
        self.assertEqual(market_structure([(1, 12)], [])["trend"], "unknown")

    def test_liquidity_sweep_bullish(self):
        n = 12
        low = [10.0] * n; high = [11.0] * n; close = [10.5] * n
        low[3] = 8.0                      # swing low at 8
        low[-1], close[-1] = 7.0, 9.0     # last candle wicks under 8 and closes back above
        df = frame(high, low, closes=close)
        h, l = swing_points(df, 3, 3)
        self.assertEqual(liquidity_sweep(df, h, l)["bullish"], 8.0)

    def test_false_breakout_up(self):
        n = 14
        high = [10.0] * n; low = [9.0] * n; close = [9.5] * n
        high[3] = 12.0                    # swing high at 12
        close[9], high[9] = 12.5, 12.8    # closes above it...
        close[-1] = 11.0                  # ...and is back below now
        df = frame(high, low, closes=close)
        h, l = swing_points(df, 3, 3)
        self.assertEqual(false_breakout(df, h, l)["up"], 12.0)

    def test_breakout_retest_confirmed(self):
        n = 16
        high = [10.0] * n; low = [9.0] * n; close = [9.5] * n; opn = [9.5] * n
        high[3] = 11.0                                   # swing high 11
        for k in (10, 11):                               # breakout bars
            close[k], high[k], low[k] = 11.6, 11.9, 11.2
        close[12], low[12], high[12], opn[12] = 11.3, 10.9, 11.5, 11.4   # dips back to level, holds
        close[13], low[13], high[13], opn[13] = 11.5, 11.0, 11.6, 11.4
        close[14], low[14], high[14], opn[14] = 11.4, 10.95, 11.5, 11.3
        close[15], low[15], high[15], opn[15] = 11.6, 11.1, 11.7, 11.3   # bullish close
        df = frame(high, low, closes=close, opens=opn)
        df["atr"], df["rel_vol"] = 0.8, 1.2
        h, l = swing_points(df, 3, 3)
        ev = detect_breakout_retest(df, h, l)["long"]
        self.assertIsNotNone(ev)
        self.assertEqual(ev["state"], "retest_confirmed")
        self.assertEqual(ev["level"], 11.0)


class RegimeTests(unittest.TestCase):
    def test_trending_vs_ranging(self):
        n = 300
        trend = pd.DataFrame(index=pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"))
        px = 100 * np.exp(0.001 * np.arange(n))
        for k, v in (("open", px), ("high", px * 1.0005), ("low", px * 0.9995), ("close", px)):
            trend[k] = v
        trend["volume"] = 10.0
        self.assertEqual(classify_regime(add_indicators(trend))["label"], "Trending")
        rng = np.random.default_rng(0)
        flat = trend.copy()
        osc = 100 + np.where(np.arange(n) % 2 == 0, 0.5, -0.5) + rng.normal(0, 0.05, n)
        for k in ("open", "high", "low", "close"):
            flat[k] = osc
        r = classify_regime(add_indicators(flat))
        self.assertEqual(r["label"], "Ranging")
        self.assertFalse(r["tradeable"])

    def test_low_volume_blocks(self):
        df = make_5m(400)
        df.iloc[-6:, df.columns.get_loc("volume")] *= 0.05
        r = classify_regime(add_indicators(df))
        self.assertTrue(r["low_volume"])
        self.assertFalse(r["tradeable"])

    def test_unknown_when_not_enough_data(self):
        self.assertEqual(classify_regime(add_indicators(make_5m(5)))["label"], "Unknown")


if __name__ == "__main__":
    unittest.main()
