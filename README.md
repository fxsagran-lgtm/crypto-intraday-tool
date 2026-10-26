# Crypto Intraday Analysis Tool (BTCUSD / ETHUSD, 5m + 15m)

**Analysis and paper trading only.** No API keys, no passwords, no order-placing code. No profit is guaranteed.

Data source: Delta Exchange India public API (BTCUSD and ETHUSD perpetual futures, USD-quoted; this is not BTC/USDT spot).

## What each file does
| File | Purpose |
|---|---|
| `config.py` | Pairs, fee/slippage defaults, all strategy thresholds |
| `data_fetcher.py` | Public candle download, retries, rate-limit handling, drops unfinished candles |
| `indicators.py` | EMA 9/21/50, RSI, ATR, daily VWAP, volume, efficiency ratio |
| `structure.py` | Swings, HH/HL/LH/LL, breakout+retest, liquidity sweeps, false breakouts |
| `regime.py` | Trending / ranging / high volatility / low volume |
| `signals.py` | 15m bias + 5m trigger + score, or NO TRADE |
| `risk.py` | Position size, fees, slippage, net reward:risk |
| `backtest.py` | Candle-by-candle backtest (no look-ahead) |
| `paper_trader.py` | Simulated trades in a local JSON file |
| `app.py` | Streamlit dashboard |
| `run_analysis.py`, `run_backtest.py`, `download_history.py`, `report.py` | Command-line tools (no Streamlit needed) |
| `test_*.py`, `synth.py` | Tests. `synth.py` makes SYNTHETIC candles and is used by tests only |

## Real data vs synthetic data
- **Real**: only from `download_history.py` (saved in `data/real/`, with a `.meta.json` stating source and time) or the live dashboard.
- **Synthetic**: only inside `synth.py` and `test_*.py`. A test fails if any production file imports it.

## Run on a computer
```
pip install -r requirements.txt
python -m unittest test_core test_engine test_data_paper     # all must say OK
python download_history.py --days 14                         # real data
python run_backtest.py --symbol BTCUSD
python run_analysis.py                                       # live analysis in text
streamlit run app.py                                         # dashboard
```

## Android phone, option A: Termux (command-line tools)
1. Install **Termux from F-Droid** (the Play Store version is outdated).
2. In Termux:
```
pkg update -y
pkg install -y python unzip
pkg install -y tur-repo
pkg install -y python-numpy python-pandas
pip install requests
termux-setup-storage
```
3. Download `intraday_tool.zip` to your phone. Then:
```
cp ~/storage/downloads/intraday_tool.zip ~
unzip intraday_tool.zip -d intraday_tool && cd intraday_tool
python -m unittest test_core test_engine test_data_paper
python download_history.py --days 14
python run_backtest.py --symbol BTCUSD
python run_analysis.py
```
The dashboard needs Streamlit, which depends on `pyarrow` and often fails to install in Termux. If `pip install streamlit plotly` fails, use option B.

## Android phone, option B: dashboard in the phone browser via GitHub Codespaces
1. Make a free GitHub account, create a new empty repository, upload all files from the zip (web "Add file > Upload files").
2. Tap **Code > Codespaces > Create codespace**.
3. In its terminal: `pip install -r requirements.txt` then `streamlit run app.py`.
4. Open the forwarded port (the "Open in Browser" pop-up) to see the dashboard.

## Honest status
See the final message of the conversation for exactly what has and has not been tested. Backtests are simulations; with fewer than 30 trades the statistics are not meaningful. Fee and slippage defaults are placeholders: check Delta's real fees.
