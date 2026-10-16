"""Position sizing: lots = (initial balance x risk%) / (stop distance x lot value), rounded DOWN.

Rounding down means the real risk is always <= the configured risk. If even the minimum lot
would risk more than allowed, the signal is blocked; it is never rounded up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import Settings


@dataclass(frozen=True)
class SizeResult:
    lots: float              # 0.0 when blocked
    risk_usd: float          # $ lost if the stop is hit at the planned price (excl. slippage)
    max_risk_usd: float      # the allowed $ risk (initial balance x risk%)
    blocked_reason: str | None = None


def position_size(stop_pts: float, settings: Settings) -> SizeResult:
    a, b = settings.account, settings.broker
    max_risk = a.initial_balance * a.risk_per_trade_pct / 100
    if not stop_pts or stop_pts <= 0 or not math.isfinite(stop_pts):
        return SizeResult(0.0, 0.0, max_risk, "invalid_stop_distance")
    raw = max_risk / (stop_pts * b.lot_value_per_point)
    # floor to the lot step; the small epsilon stops 0.3 / 0.01 = 29.999... from losing a step
    steps = math.floor(raw / b.lot_step + 1e-9)
    lots = round(steps * b.lot_step, 8)
    if lots < b.min_lot:
        return SizeResult(0.0, 0.0, max_risk, "below_min_lot")
    risk = lots * stop_pts * b.lot_value_per_point
    if risk > max_risk + 1e-9:   # cannot happen after flooring; kept as a hard guard
        return SizeResult(0.0, 0.0, max_risk, "risk_above_limit")
    return SizeResult(lots, risk, max_risk)
