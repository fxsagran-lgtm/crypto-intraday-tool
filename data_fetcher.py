"""Public market-data client for Delta Exchange India. No API key is used or needed."""
import time
from typing import Callable, Optional

import pandas as pd
import requests

from config import BASE_URL, CANDLES_PATH, TF_SECONDS

CHUNK_BARS = 500  # candles per request (kept conservatively small)


class DataSourceError(Exception):
    """Raised for any problem getting or parsing market data."""


def parse_candles(payload: dict) -> pd.DataFrame:
    """Convert the API JSON into a UTC-indexed OHLCV DataFrame (index = candle OPEN time)."""
    if not isinstance(payload, dict) or "result" not in payload:
        raise DataSourceError(f"Unexpected response format: {str(payload)[:200]}")
    if payload.get("success") is False:
        raise DataSourceError(f"API reported failure: {str(payload)[:200]}")
    rows = payload["result"]
    cols = ["time", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=cols[1:], index=pd.DatetimeIndex([], tz="UTC", name="time"))
    try:
        df = pd.DataFrame(rows)[cols]
    except KeyError as exc:
        raise DataSourceError(f"Candle rows are missing field {exc}") from exc
    df["time"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="s", utc=True)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().set_index("time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def count_gaps(df: pd.DataFrame, tf_seconds: int) -> int:
    """Number of missing candles inside the fetched range."""
    if len(df) < 2:
        return 0
    steps = df.index.to_series().diff().dropna().dt.total_seconds()
    return int(((steps / tf_seconds).round() - 1).clip(lower=0).sum())


class DeltaClient:
    def __init__(self, base_url: str = BASE_URL, session: Optional[requests.Session] = None,
                 min_interval: float = 0.25, max_retries: int = 4, timeout: float = 10.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time):
        self.base_url, self.session = base_url, session or requests.Session()
        self.min_interval, self.max_retries, self.timeout = min_interval, max_retries, timeout
        self._sleep, self._clock, self._last = sleep, clock, 0.0

    def _get(self, params: dict) -> dict:
        url, last_err = self.base_url + CANDLES_PATH, "unknown error"
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = self._clock()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_err = f"network error: {exc}"
                self._sleep(min(2 ** attempt, 15))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code} (rate limit / server)"
                retry_after = resp.headers.get("Retry-After") if getattr(resp, "headers", None) else None
                try:
                    delay = float(retry_after) if retry_after else min(2 ** attempt, 15)
                except ValueError:
                    delay = min(2 ** attempt, 15)
                self._sleep(delay)
                continue
            if resp.status_code != 200:
                raise DataSourceError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            try:
                return resp.json()
            except ValueError as exc:
                raise DataSourceError("Response was not valid JSON") from exc
        raise DataSourceError(f"Gave up after {self.max_retries + 1} attempts: {last_err}")

    def fetch(self, symbol: str, resolution: str, n_bars: int, end: Optional[int] = None,
              drop_incomplete: bool = True) -> pd.DataFrame:
        """Return the most recent `n_bars` CLOSED candles ending at `end` (epoch seconds, default now)."""
        if resolution not in TF_SECONDS:
            raise ValueError(f"Unsupported resolution {resolution}; use one of {list(TF_SECONDS)}")
        tf = TF_SECONDS[resolution]
        now = int(self._clock())
        end = int(end or now)
        frames, remaining, chunk_end = [], n_bars, end
        while remaining > 0:
            take = min(remaining, CHUNK_BARS)
            start = chunk_end - take * tf
            payload = self._get({"resolution": resolution, "symbol": symbol, "start": start, "end": chunk_end})
            part = parse_candles(payload)
            if part.empty:
                break
            frames.append(part)
            remaining -= take
            chunk_end = start - 1
        if not frames:
            raise DataSourceError(f"No candles returned for {symbol} {resolution}")
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if drop_incomplete:  # a candle is closed only once open_time + timeframe <= now
            df = df[df.index + pd.Timedelta(seconds=tf) <= pd.Timestamp(now, unit="s", tz="UTC")]
        return df.tail(n_bars)
