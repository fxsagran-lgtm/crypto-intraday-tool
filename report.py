"""Plain-text rendering shared by the command-line tools."""
from signals import TradeSignal


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def render_signal(s: TradeSignal, data_note: str = "") -> str:
    L = [f"=== {s.symbol} | {s.action} ===", data_note,
         f"Latest closed 5m candle: {s.candle_time}",
         f"15m: structure={s.structure_15m}, bias={s.bias.upper()} (score {s.bias_score:+d}), "
         f"regime={s.regime_15m.get('label', 'n/a')}",
         f"5m : structure={s.structure_5m}, regime={s.regime_5m.get('label', 'n/a')}",
         f"Timeframes: {s.timeframe_agreement or 'n/a'}"]
    ind = s.indicators.get("5m")
    if ind:
        L.append("5m indicators: " + ", ".join(f"{k}={v:,.2f}" for k, v in ind.items()))
    if s.action == "NO TRADE":
        L.append("Why no trade:")
        L += [f"  - {r}" for r in s.no_trade_reasons] or ["  - (no reason recorded)"]
    else:
        L += [f"Entry zone : {fmt(s.entry_low)} - {fmt(s.entry_high)}",
              f"Stop-loss  : {fmt(s.stop)}",
              "Targets    : " + " | ".join(f"T{i + 1} {fmt(t)} ({r:.0f}R)" for i, (t, r) in enumerate(zip(s.targets, s.rr_gross))),
              f"R:R after fees/slippage to T2: {s.rr_net_t2:.2f}",
              f"Score      : {s.score}/{s.max_score}", "Reasons:"]
        L += [f"  + {r}" for r in s.reasons]
        L.append("Before entering:")
        L += [f"  * {c}" for c in s.conditions]
        L.append(f"Invalidation: {s.invalidation}")
    return "\n".join(x for x in L if x != "")
