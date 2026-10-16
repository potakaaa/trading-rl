"""Bar-by-bar 1-minute order simulation for one setup. Pure function: no account state.

Price data is BID. Buys fill at ASK = BID + spread, sells at BID:
  long : buy-limit entry fills when ask low <= entry; exits are sells at bid
  short: sell-limit entry fills when bid high >= entry; exits are buys at ask
Pessimistic rules (spec section 6):
  * if stop and target are both touched in the same 1m bar, the stop is assumed hit first
  * on the fill bar only the stop is checked (we cannot know if the target came before the fill)
  * stops fill `stop_slippage` points worse; a bar that opens through the stop exits at the open
  * limit entries fill at the limit price, never better
Only bars with ts >= armed_at are looked at, so nothing before the signal can leak in, and the
simulation stops at the exit, so nothing after the exit is used either.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class M1:
    """1-minute BID bars as numpy arrays for fast scanning. ts = bar start (UTC, int64 ns)."""
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> M1:
        # parquet round-trips can give ms/us resolution; all comparisons below are in ns
        idx = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
        return cls(idx.asi8, *(df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")))

    def first_at_or_after(self, t: pd.Timestamp) -> int:
        return int(np.searchsorted(self.ts, pd.Timestamp(t).value, side="left"))


@dataclass(frozen=True)
class Costs:
    spread: float
    stop_slippage: float
    commission_per_lot: float = 0.0


@dataclass
class Fill:
    outcome: str                 # tp | sl | session_end | news_flat | expired | news_cancel | no_data
    fill_time: pd.Timestamp | None = None
    exit_time: pd.Timestamp | None = None
    fill_price: float = np.nan
    exit_price: float = np.nan
    pnl_pts: float = 0.0         # per 1.0 lot, before commission
    mfe_pts: float = 0.0         # best favourable move before the exit (excl. the exit bar if stopped)
    mae_pts: float = 0.0         # worst adverse move while open (for intraday equity checks)

    @property
    def filled(self) -> bool:
        return self.fill_time is not None


def _t(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, tz="UTC")


def simulate(m1: M1, direction: str, entry: float, stop: float, target: float,
             armed_at: pd.Timestamp, order_deadline: pd.Timestamp, flat_deadline: pd.Timestamp,
             costs: Costs, deadline_reason: str = "expired", flat_reason: str = "session_end",
             order_type: str = "limit") -> Fill:
    """Simulate a limit order (or a market order, `order_type="market"`) placed at `armed_at`.

    Market orders fill at the OPEN of the first 1m bar at/after armed_at (longs at the ask), so the
    whole fill bar happens after the fill: its target counts (stop still wins a tie). `entry` is then
    ignored and the actual fill price is returned in Fill.fill_price.

    order_deadline: bars starting at/after this cannot fill (order expiry, news cut, session end).
    flat_deadline : an open position is closed at the close of the last bar starting before it.
    """
    short = direction == "short"
    sp, slip = costs.spread, costs.stop_slippage
    i = m1.first_at_or_after(armed_at)
    od, fd = pd.Timestamp(order_deadline).value, pd.Timestamp(flat_deadline).value
    n = len(m1.ts)
    if i >= n or m1.ts[i] >= od:
        return Fill("no_data" if i >= n else deadline_reason)

    market = order_type == "market"
    fill_i = -1
    if market:
        fill_i = i
        entry = m1.open[i] if short else m1.open[i] + sp
    else:
        # ------------------------------------------------------------ wait for the limit fill
        while i < n and m1.ts[i] < od:
            if (m1.high[i] >= entry) if short else (m1.low[i] + sp <= entry):
                fill_i = i
                break
            i += 1
    if fill_i < 0:
        return Fill(deadline_reason)

    f = Fill("open", fill_time=_t(m1.ts[fill_i]), fill_price=entry)
    mfe = mae = 0.0
    j = fill_i
    while j < n and m1.ts[j] < fd:
        o, h, lo = m1.open[j], m1.high[j], m1.low[j]
        if short:
            ask_o, ask_h, ask_l = o + sp, h + sp, lo + sp
            stop_hit, tgt_hit = ask_h >= stop, ask_l <= target
            adverse, favour = ask_h - entry, entry - ask_l
        else:
            stop_hit, tgt_hit = lo <= stop, h >= target
            adverse, favour = entry - lo, h - entry
        if stop_hit:
            if j == fill_i:
                px = stop + slip if short else stop - slip
            elif short:
                px = max(stop, ask_o) + slip      # gapped through the stop -> worse fill
            else:
                px = min(stop, o) - slip
            # the position is gone at the stop: the worst equity is the realized exit, not the bar extreme
            mae = max(mae, (px - entry) if short else (entry - px))
            f.outcome, f.exit_time, f.exit_price = "sl", _t(m1.ts[j] + 60_000_000_000), px
            break
        mae = max(mae, adverse)
        after_fill = market or j > fill_i         # a limit fill bar's extreme may come before the fill
        if after_fill:
            mfe = max(mfe, favour)
        if tgt_hit and after_fill:
            f.outcome, f.exit_time, f.exit_price = "tp", _t(m1.ts[j] + 60_000_000_000), target
            break
        j += 1
    if f.outcome == "open":
        last = min(j, n) - 1                       # last bar that started before the deadline
        if last < fill_i:
            last = fill_i
        c = m1.close[last]
        f.outcome = flat_reason
        f.exit_time = _t(m1.ts[last] + 60_000_000_000)
        f.exit_price = c + sp if short else c
    f.pnl_pts = (entry - f.exit_price) if short else (f.exit_price - entry)
    f.mfe_pts, f.mae_pts = mfe, mae
    return f
