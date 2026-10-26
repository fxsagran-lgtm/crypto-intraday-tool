"""Position sizing and cost-aware reward:risk. All numbers are per 1 unit of the base asset."""
from typing import Optional


def _legs(entry: float, stop: float, side: str, slip: float):
    if side == "long":
        return entry * (1 + slip), stop * (1 - slip)   # worst-case fill in, stop-market fill out
    return entry * (1 - slip), stop * (1 + slip)


def loss_per_unit_at_stop(entry: float, stop: float, side: str, fee: float, slip: float) -> float:
    eff_in, eff_out = _legs(entry, stop, side, slip)
    gross = (eff_in - eff_out) if side == "long" else (eff_out - eff_in)
    return gross + fee * (eff_in + eff_out)


def position_size(balance: float, risk_pct: float, entry: float, stop: float, side: str,
                  fee: float, slip: float, max_leverage: Optional[float] = None) -> dict:
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if balance <= 0 or not (0 < risk_pct <= 100):
        raise ValueError("balance must be > 0 and risk_pct in (0, 100]")
    per_unit = loss_per_unit_at_stop(entry, stop, side, fee, slip)
    if per_unit <= 0:
        raise ValueError("stop-loss is on the wrong side of entry")
    risk_amount = balance * risk_pct / 100.0
    qty = risk_amount / per_unit
    capped = False
    if max_leverage and qty * entry > balance * max_leverage:
        qty, capped = balance * max_leverage / entry, True
    notional = qty * entry
    return {"qty": qty, "notional": notional, "risk_amount_planned": risk_amount,
            "risk_amount_at_stop": qty * per_unit, "leverage_used": notional / balance,
            "capped_by_leverage": capped}


def net_reward_risk(entry: float, stop: float, target: float, side: str, fee: float, slip: float) -> float:
    """Reward:risk after fees and slippage (target assumed a limit fill: no exit slippage)."""
    risk_u = loss_per_unit_at_stop(entry, stop, side, fee, slip)
    eff_in = entry * (1 + slip) if side == "long" else entry * (1 - slip)
    gross = (target - eff_in) if side == "long" else (eff_in - target)
    reward_u = gross - fee * (eff_in + target)
    return reward_u / risk_u if risk_u > 0 else float("nan")
