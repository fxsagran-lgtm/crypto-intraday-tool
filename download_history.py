"""Download REAL historical candles from Delta Exchange India's public API into data/real/*.csv.
Run this on your own device (it needs internet). No account or credentials are used.

    python download_history.py --days 14
"""
import argparse
import json
import os
from datetime import datetime, timezone

from config import DATA_SOURCE_NAME, SYMBOLS, TF_SECONDS
from data_fetcher import DeltaClient, DataSourceError, count_gaps

OUT_DIR = os.path.join("data", "real")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="history length (default 14)")
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS.values()))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    client = DeltaClient()
    for sym in args.symbols:
        for res, tf in TF_SECONDS.items():
            n = args.days * 86400 // tf
            try:
                df = client.fetch(sym, res, n)
            except DataSourceError as exc:
                print(f"FAILED {sym} {res}: {exc}")
                return 1
            path = os.path.join(OUT_DIR, f"{sym}_{res}.csv")
            df.to_csv(path)
            meta = {"source": DATA_SOURCE_NAME, "real_market_data": True, "symbol": sym, "resolution": res,
                    "candles": len(df), "first_candle": str(df.index[0]), "last_candle": str(df.index[-1]),
                    "missing_candles": count_gaps(df, tf),
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat()}
            with open(path + ".meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
            print(f"OK {sym} {res}: {len(df)} candles {meta['first_candle']} -> {meta['last_candle']} "
                  f"(missing: {meta['missing_candles']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
