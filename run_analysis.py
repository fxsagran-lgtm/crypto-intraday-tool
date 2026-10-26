"""Command-line analysis (works anywhere Python runs, including a phone terminal). Analysis only."""
import argparse
from datetime import datetime, timezone

from config import DATA_SOURCE_NAME, Params, SYMBOLS
from data_fetcher import DataSourceError, DeltaClient
from report import render_signal
from signals import analyze


def main() -> int:
    ap = argparse.ArgumentParser(description="Live analysis of BTC and ETH (no orders are ever placed).")
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS.values()))
    args = ap.parse_args()
    client, p = DeltaClient(), Params()
    for sym in args.symbols:
        try:
            d5, d15 = client.fetch(sym, "5m", 400), client.fetch(sym, "15m", 250)
        except DataSourceError as exc:
            print(f"{sym}: could not get data -> {exc}")
            continue
        note = (f"Data: {DATA_SOURCE_NAME} | fetched {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC "
                f"| last 5m candle {d5.index[-1]:%H:%M} UTC")
        print(render_signal(analyze(sym, d5, d15, p), note), "\n")
    print("Analysis only. Not financial advice. No profit is guaranteed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
