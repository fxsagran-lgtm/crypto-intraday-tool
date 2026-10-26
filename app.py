"""Streamlit dashboard. ANALYSIS + PAPER TRADING ONLY. No orders are ever sent to any exchange."""
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import run_backtest
from config import DATA_SOURCE_NAME, Params, SYMBOLS
from data_fetcher import DataSourceError, DeltaClient, count_gaps
from indicators import add_indicators
from paper_trader import PaperTrader
from risk import position_size
from signals import analyze

st.set_page_config(page_title="Crypto Intraday Analysis", layout="centered")
st.title("Crypto Intraday Analysis")
st.caption("Analysis and paper trading only. No real orders. Not financial advice; no profit is guaranteed.")

with st.sidebar:
    st.header("Risk settings")
    balance = st.number_input("Account balance (USD)", min_value=10.0, value=1000.0, step=100.0)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    max_lev = st.number_input("Max leverage (x)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
    fee_pct = st.number_input("Fee per side (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                              help="Placeholder. Check Delta's real fee schedule.")
    slip_pct = st.number_input("Slippage per fill (%)", min_value=0.0, max_value=1.0, value=0.03, step=0.01)
    min_score = st.slider("Minimum setup score (of 11)", 4, 10, 6)
    if st.button("Refresh data now"):
        st.cache_data.clear()

P = replace(Params(), fee_rate=fee_pct / 100, slippage_rate=slip_pct / 100, min_score=min_score)


@st.cache_data(ttl=20, show_spinner=False)
def load_live(symbol: str):
    c = DeltaClient()
    return c.fetch(symbol, "5m", 400), c.fetch(symbol, "15m", 250), datetime.now(timezone.utc)


@st.cache_data(ttl=600, show_spinner=False)
def load_history(symbol: str, days: int):
    c = DeltaClient()
    return c.fetch(symbol, "5m", days * 288), c.fetch(symbol, "15m", days * 96 + 120), datetime.now(timezone.utc)


def chart(d5: pd.DataFrame, sig) -> go.Figure:
    x = add_indicators(d5, P).tail(120)
    fig = go.Figure(go.Candlestick(x=x.index, open=x["open"], high=x["high"], low=x["low"], close=x["close"], name="5m"))
    for col, name in (("ema9", "EMA 9"), ("ema21", "EMA 21"), ("ema50", "EMA 50"), ("vwap", "VWAP")):
        fig.add_trace(go.Scatter(x=x.index, y=x[col], name=name, line=dict(width=1)))
    if sig.action != "NO TRADE":
        fig.add_hline(y=sig.entry, line_dash="dot", annotation_text="Entry")
        fig.add_hline(y=sig.stop, line_color="red", annotation_text="Stop")
        for i, t in enumerate(sig.targets, 1):
            fig.add_hline(y=t, line_color="green", annotation_text=f"T{i}")
    fig.update_layout(height=380, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h"))
    return fig


def show_signal(sym: str, sig, d5, fetched):
    st.subheader(sym)
    st.caption(f"Source: {DATA_SOURCE_NAME} | fetched {fetched:%Y-%m-%d %H:%M:%S} UTC | "
               f"last closed 5m candle: {sig.candle_time:%Y-%m-%d %H:%M} UTC")
    if sig.action == "LONG":
        st.success("LONG setup")
    elif sig.action == "SHORT":
        st.error("SHORT setup")
    else:
        st.warning("NO TRADE: no valid setup right now")
    c1, c2, c3 = st.columns(3)
    c1.metric("15m bias", sig.bias.upper(), f"score {sig.bias_score:+d}")
    c2.metric("15m market", sig.regime_15m.get("label", "n/a"))
    c3.metric("5m market", sig.regime_5m.get("label", "n/a"))
    st.write(f"**Timeframes:** {sig.timeframe_agreement or 'n/a'}")
    r15, r5 = sig.regime_15m, sig.regime_5m
    flags = [f"15m {k}" for k, v in (("trending", r15.get("trending")), ("ranging", r15.get("ranging"))) if v]
    flags += [f"5m {k}" for k, v in (("high volatility", r5.get("high_volatility")), ("low volume", r5.get("low_volume"))) if v]
    fb = sig.flags.get("false_breakout", {}) if sig.flags else {}
    if fb.get("up") or fb.get("down"):
        flags.append("possible false breakout")
    st.write("**Market conditions:** " + (", ".join(flags) if flags else "nothing unusual flagged"))
    if sig.action == "NO TRADE":
        st.write("**Why no trade:**")
        for r in sig.no_trade_reasons:
            st.write(f"- {r}")
    else:
        st.table(pd.DataFrame({
            "Level": ["Entry zone", "Stop-loss", "Target 1", "Target 2", "Target 3"],
            "Price": [f"{sig.entry_low:,.2f} - {sig.entry_high:,.2f}", f"{sig.stop:,.2f}",
                      *[f"{t:,.2f}" for t in sig.targets]],
            "R:R": ["", "", "1 : 1", "1 : 2", "1 : 3"]}))
        st.write(f"**Reward:risk to Target 2 after fees & slippage:** {sig.rr_net_t2:.2f}  |  **Score:** {sig.score}/{sig.max_score}")
        side = sig.action.lower()
        try:
            sz = position_size(balance, risk_pct, sig.entry, sig.stop, side, P.fee_rate, P.slippage_rate, max_lev)
            st.write(f"**Position size:** {sz['qty']:.5f} {sym[:3]} (about ${sz['notional']:,.0f}, "
                     f"{sz['leverage_used']:.1f}x). Loss if stopped out, including costs: ${sz['risk_amount_at_stop']:.2f}"
                     + (" (size capped by leverage limit)" if sz["capped_by_leverage"] else ""))
        except ValueError as exc:
            st.error(f"Could not size position: {exc}")
        st.write("**Reasons:**")
        for r in sig.reasons:
            st.write(f"- {r}")
        st.write("**Conditions before entry:**")
        for c in sig.conditions:
            st.write(f"- {c}")
        st.write(f"**Invalidation:** {sig.invalidation}")
        pt = PaperTrader(start_balance=balance)
        if st.button(f"Paper-trade this {sym} setup", key=f"paper_{sym}"):
            try:
                pt.open_from_signal(sig, risk_pct, P, max_lev)
                st.success("Paper position opened (simulated only).")
            except ValueError as exc:
                st.error(str(exc))
    if sig.indicators:
        ind = pd.DataFrame(sig.indicators).T.rename(columns={"atr": "ATR", "rsi": "RSI", "rel_vol": "Rel. volume",
                                                              "er": "Efficiency", "ema9": "EMA 9", "ema21": "EMA 21",
                                                              "ema50": "EMA 50", "vwap": "VWAP", "close": "Close"})
        st.dataframe(ind.round(2), use_container_width=True)
    st.plotly_chart(chart(d5, sig), use_container_width=True)


tab_live, tab_bt, tab_paper, tab_about = st.tabs(["Live analysis", "Backtest", "Paper trading", "About"])

with tab_live:
    for name, sym in SYMBOLS.items():
        try:
            with st.spinner(f"Loading {sym}..."):
                d5, d15, fetched = load_live(sym)
            gaps = count_gaps(d5, 300)
            if gaps:
                st.warning(f"{sym}: {gaps} missing 5m candles in the fetched window; indicators may be slightly off.")
            show_signal(sym, analyze(sym, d5, d15, P), d5, fetched)
        except DataSourceError as exc:
            st.error(f"{sym}: could not load market data ({exc}). Try again in a minute.")
        st.divider()

with tab_bt:
    st.markdown("**BACKTEST = simulation on past candles. It is not live performance and does not predict future results.**")
    sym = st.selectbox("Pair", list(SYMBOLS.values()))
    days = st.slider("Days of history", 3, 30, 14)
    tgt = st.selectbox("Exit target", [1.0, 2.0, 3.0], index=1, format_func=lambda v: f"{v:.0f}R")
    if st.button("Run backtest on real data"):
        try:
            with st.spinner("Downloading history and simulating (can take up to a minute)..."):
                h5, h15, fetched = load_history(sym, days)
                res = run_backtest(sym, h5, h15, replace(P, target_r=tgt), balance, risk_pct, max_lev)
            s = res.stats
            st.info(res.label)
            st.caption(f"Source: {DATA_SOURCE_NAME} | {h5.index[0]:%Y-%m-%d} to {h5.index[-1]:%Y-%m-%d} UTC | "
                       f"fee {fee_pct}%/side, slippage {slip_pct}%")
            a, b, c = st.columns(3)
            a.metric("Trades", s["total_trades"])
            b.metric("Win rate", "n/a" if s["win_rate_pct"] is None else f"{s['win_rate_pct']:.0f}%")
            c.metric("Profit factor", "n/a" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}")
            d, e = st.columns(2)
            d.metric("Net return", f"{s['net_return_pct']:.2f}%")
            e.metric("Max drawdown", f"{s['max_drawdown_pct']:.2f}%")
            if s["low_sample"]:
                st.warning("Fewer than 30 trades: these numbers are not reliable evidence of anything.")
            if len(res.trades):
                st.line_chart(res.equity)
                st.dataframe(res.trades.round({c: 4 for c in res.trades.select_dtypes('number').columns}), use_container_width=True)
        except DataSourceError as exc:
            st.error(f"Could not download history: {exc}")

with tab_paper:
    pt = PaperTrader(start_balance=balance)
    st.write(f"**Paper balance:** ${pt.state['balance']:,.2f}  (started at ${pt.state['start_balance']:,.2f}). Simulated money only.")
    if st.button("Check open paper trades against latest candles"):
        try:
            for sym in SYMBOLS.values():
                d5, _, _ = load_live(sym)
                for t in pt.update(sym, d5, P):
                    st.info(f"{t['symbol']} closed by {t['exit_reason']}: P&L ${t['pnl']:.2f} ({t['r_multiple']:.2f}R)")
        except DataSourceError as exc:
            st.error(str(exc))
    if pt.state["open"]:
        st.write("Open paper positions"); st.dataframe(pd.DataFrame(pt.state["open"]))
    if pt.state["closed"]:
        st.write("Closed paper trades"); st.dataframe(pd.DataFrame(pt.state["closed"]))
    if not pt.state["open"] and not pt.state["closed"]:
        st.write("No paper trades yet.")

with tab_about:
    st.markdown(f"""
**Data:** {DATA_SOURCE_NAME}. Delta India lists BTC and ETH as USD-quoted perpetual futures (BTCUSD, ETHUSD), not BTC/USDT spot.

**Strategy:** the 15m chart sets the bias; the 5m chart supplies entries (EMA-21 pullbacks and breakout-retests)
and a score out of 11 must reach your minimum. Stops sit beyond a structure level plus an ATR buffer; targets are 1R, 2R and 3R.
Ranging, thin-volume, over-volatile or conflicting conditions give NO TRADE.

**Safety:** no credentials and no order endpoints exist in this code. Paper trades are written to a local file only.
""")
