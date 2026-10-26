"""Tests for the data client (mocked HTTP, no network), paper trader, and data separation."""
import os
import re
import tempfile
import unittest

import pandas as pd
import requests

from config import Params
from data_fetcher import DataSourceError, DeltaClient, count_gaps, parse_candles
from paper_trader import PaperTrader
from signals import TradeSignal


class Resp:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code, self._d, self.headers, self.text = status, data, headers or {}, str(data)
    def json(self):
        if self._d is None:
            raise ValueError("no json")
        return self._d


def candles(start, n, tf=300):
    return [{"time": start + k * tf, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 3.0} for k in range(n)]


class FakeSession:
    """Serves candles for any requested [start, end] window, like the real endpoint."""
    def __init__(self, script=None):
        self.calls, self.script = [], list(script or [])
    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        tf = {"5m": 300, "15m": 900}[params["resolution"]]
        first = -(-params["start"] // tf) * tf
        rows = [{"time": t, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 3}
                for t in range(first, params["end"] + 1, tf)]
        return Resp(200, {"success": True, "result": list(reversed(rows))})   # newest first, like many APIs


def client(session, now=1_800_000_000):
    sleeps = []
    c = DeltaClient(session=session, min_interval=0, max_retries=3, sleep=sleeps.append, clock=lambda: now)
    c.sleeps = sleeps
    return c


class ParseTests(unittest.TestCase):
    def test_parse_sorts_and_types(self):
        df = parse_candles({"success": True, "result": list(reversed(candles(1_700_000_000, 3)))})
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertEqual(str(df.index.tz), "UTC")
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])

    def test_bad_payloads(self):
        with self.assertRaises(DataSourceError):
            parse_candles({"oops": 1})
        with self.assertRaises(DataSourceError):
            parse_candles({"success": False, "result": []})
        with self.assertRaises(DataSourceError):
            parse_candles({"success": True, "result": [{"time": 1, "open": 1}]})
        self.assertTrue(parse_candles({"success": True, "result": []}).empty)

    def test_count_gaps(self):
        df = parse_candles({"result": candles(1_700_000_000, 10)})
        self.assertEqual(count_gaps(df, 300), 0)
        self.assertEqual(count_gaps(df.drop(df.index[4]), 300), 1)


class FetchTests(unittest.TestCase):
    def test_pagination_returns_exact_count_sorted_unique(self):
        s = FakeSession(); c = client(s)
        df = c.fetch("BTCUSD", "5m", 1200)
        self.assertEqual(len(df), 1200)
        self.assertTrue(df.index.is_monotonic_increasing and df.index.is_unique)
        self.assertGreaterEqual(len(s.calls), 3)
        self.assertEqual(count_gaps(df, 300), 0)

    def test_unfinished_candle_is_dropped(self):
        now = 1_800_000_100                       # 100s into a 5m candle
        df = client(FakeSession(), now).fetch("BTCUSD", "5m", 50)
        self.assertLessEqual(df.index[-1].timestamp() + 300, now)
        df2 = client(FakeSession(), now).fetch("BTCUSD", "5m", 50, drop_incomplete=False)
        self.assertGreater(df2.index[-1], df.index[-1])

    def test_retries_on_429_then_succeeds(self):
        ok = Resp(200, {"success": True, "result": candles(1_799_990_000, 5)})
        s = FakeSession([Resp(429, {}, {"Retry-After": "2"}), Resp(503, {}), ok]); c = client(s)
        df = c.fetch("BTCUSD", "5m", 5, drop_incomplete=False)
        self.assertEqual(len(df), 5)
        self.assertEqual(c.sleeps[0], 2.0)

    def test_gives_up_after_retries(self):
        s = FakeSession([Resp(500, {})] * 10)
        with self.assertRaises(DataSourceError):
            client(s).fetch("BTCUSD", "5m", 5)

    def test_network_errors_are_handled(self):
        ok = Resp(200, {"success": True, "result": candles(1_799_990_000, 5)})
        s = FakeSession([requests.Timeout("slow"), requests.ConnectionError("down"), ok])
        self.assertEqual(len(client(s).fetch("BTCUSD", "5m", 5, drop_incomplete=False)), 5)

    def test_client_error_and_bad_json_raise(self):
        with self.assertRaises(DataSourceError):
            client(FakeSession([Resp(404, {})])).fetch("NOPE", "5m", 5)
        with self.assertRaises(DataSourceError):
            client(FakeSession([Resp(200, None)])).fetch("BTCUSD", "5m", 5)

    def test_unsupported_resolution(self):
        with self.assertRaises(ValueError):
            client(FakeSession()).fetch("BTCUSD", "1h", 5)


def long_signal():
    return TradeSignal("BTCUSD", action="LONG", entry=100.0, stop=98.0, score=8,
                       candle_time=pd.Timestamp("2026-01-01 10:00", tz="UTC"))


def bars(rows, start="2026-01-01 10:05"):
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "paper.json")

    def test_target_hit_and_persistence(self):
        pt = PaperTrader(self.path, 1000.0)
        pt.open_from_signal(long_signal(), 1.0)
        done = pt.update("BTCUSD", bars([[100, 101, 99.5, 100.5], [100.5, 104.5, 100, 104]]))
        self.assertEqual(done[0]["exit_reason"], "target")
        self.assertGreater(pt.state["balance"], 1000.0)
        again = PaperTrader(self.path)
        self.assertEqual(again.state["balance"], pt.state["balance"])
        self.assertEqual(len(again.state["closed"]), 1)

    def test_stop_hit_loses_about_one_percent(self):
        pt = PaperTrader(self.path, 1000.0)
        pt.open_from_signal(long_signal(), 1.0)
        done = pt.update("BTCUSD", bars([[100, 100.5, 97.0, 97.5]]))
        self.assertEqual(done[0]["exit_reason"], "stop")
        self.assertAlmostEqual(pt.state["balance"], 990.0, places=0)

    def test_still_open_when_nothing_hit(self):
        pt = PaperTrader(self.path, 1000.0)
        pt.open_from_signal(long_signal(), 1.0)
        self.assertEqual(pt.update("BTCUSD", bars([[100, 101, 99.5, 100.5]])), [])
        self.assertTrue(pt.has_open("BTCUSD"))

    def test_refuses_no_trade_and_duplicates(self):
        pt = PaperTrader(self.path, 1000.0)
        with self.assertRaises(ValueError):
            pt.open_from_signal(TradeSignal("BTCUSD"), 1.0)
        pt.open_from_signal(long_signal(), 1.0)
        with self.assertRaises(ValueError):
            pt.open_from_signal(long_signal(), 1.0)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class SeparationTests(unittest.TestCase):
    def test_production_modules_never_import_synthetic_data(self):
        for f in ("app.py", "run_analysis.py", "run_backtest.py", "download_history.py", "backtest.py",
                  "signals.py", "data_fetcher.py", "paper_trader.py"):
            if os.path.exists(f):
                src = read(f)
                self.assertIsNone(re.search(r"^\s*(import|from)\s+synth\b", src, re.M), f"{f} imports synth")

    def test_no_secrets_or_order_endpoints_in_code(self):
        for f in os.listdir("."):
            if f.endswith(".py") and not f.startswith("test_"):
                src = read(f).lower()
                for bad in ("api_secret", "api-key", "/v2/orders", "place_order", "password"):
                    self.assertNotIn(bad, src, f"{f} contains '{bad}'")


if __name__ == "__main__":
    unittest.main()
