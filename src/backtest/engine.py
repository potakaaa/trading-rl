"""Event-driven backtest over the setups produced by `detect_all` (the same code the live engine uses).

Two runs:
  raw     : every valid setup traded independently, costs included, no prop/news rules.
            This answers "do the rule-based setups have an edge after costs?" (the Phase 3 gate).
  guarded : chronological, one account, every signal passes `risk.guard.evaluate` exactly as live
            (news blackout from the historical calendar, daily/max-drawdown buffers, signal caps,
            consecutive losses, sizing). Positions are flattened at session end and 1 minute before
            a news blackout starts; pending orders are cancelled at the same moment.
Higher-timeframe context is already lookahead-free inside detect_all; here only 1m bars at or
after each signal time are read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.fills import M1, Costs, Fill, simulate
from src.config import Settings
from src.news.calendar import NewsCalendar
from src.risk.guard import AccountState, Limits, SignalCandidate, evaluate
from src.risk.sizing import position_size
from src.timeutils import session_end_utc, trading_date

TRADE_COLUMNS = [
    "setup_id", "setup_type", "direction", "session", "armed_at", "expires_at", "entry", "stop",
    "target", "rr", "risk_pts", "outcome", "filled", "fill_time", "exit_time", "fill_price",
    "exit_price", "pnl_pts", "r", "mfe_r", "mae_r", "hit_1r", "hit_2r", "hit_3r", "lots",
    "risk_usd", "pnl_usd", "balance_after", "trading_day",
]


def _order_type(s) -> str:
    v = s.get("order_type") if hasattr(s, "get") else None
    return v if isinstance(v, str) else "limit"


def costs_from_settings(s: Settings, spread_mult: float = 1.0) -> Costs:
    return Costs(spread=s.broker.typical_spread_points * spread_mult,
                 stop_slippage=s.backtest.stop_slippage_points,
                 commission_per_lot=s.backtest.commission_per_lot)


def next_blackout_start(cal: NewsCalendar | None, t: pd.Timestamp) -> pd.Timestamp | None:
    """Start of the first blackout window that begins after t (None if no later event)."""
    if cal is None or len(cal._ts) == 0:
        return None
    starts = cal._ts - cal.before.to_timedelta64()
    k = int(np.searchsorted(starts, pd.Timestamp(t).tz_convert("UTC").tz_localize(None).to_datetime64(),
                            side="right"))
    return pd.Timestamp(starts[k]).tz_localize("UTC") if k < len(starts) else None


def _deadlines(setup, settings: Settings, cal: NewsCalendar | None):
    armed = pd.Timestamp(setup["armed_at"])
    s_end = session_end_utc(armed, settings.sessions[setup["session"]].end)
    order_dl, flat_dl = min(pd.Timestamp(setup["expires_at"]), s_end), s_end
    order_reason, flat_reason = "expired", "session_end"
    nb = next_blackout_start(cal, armed)
    if nb is not None:
        cut = nb - pd.Timedelta(minutes=1)
        if cut < order_dl:
            order_dl, order_reason = cut, "news_cancel"
        if cut < flat_dl:
            flat_dl, flat_reason = cut, "news_flat"
    return armed, order_dl, flat_dl, order_reason, flat_reason


def _trade_row(sid, setup, f: Fill, costs: Costs, lot_value: float, lots: float | None) -> dict:
    risk_pts = float(setup["risk_pts"])
    if setup.get("order_type") == "market" and f.filled:
        risk_pts = abs(f.fill_price - float(setup["stop"]))        # R is measured from the actual fill
        if risk_pts <= 0:
            risk_pts = float(setup["risk_pts"])
    comm_r = costs.commission_per_lot / (risk_pts * lot_value) if f.filled else 0.0
    r = f.pnl_pts / risk_pts - comm_r if f.filled else 0.0
    mfe_r = f.mfe_pts / risk_pts if f.filled else 0.0
    row = {
        "setup_id": sid, "setup_type": setup["setup_type"], "direction": setup["direction"],
        "session": setup["session"], "armed_at": setup["armed_at"], "expires_at": setup["expires_at"],
        "entry": setup["entry"], "stop": setup["stop"], "target": setup["target"], "rr": setup["rr"],
        "risk_pts": risk_pts, "outcome": f.outcome, "filled": f.filled, "fill_time": f.fill_time,
        "exit_time": f.exit_time, "fill_price": f.fill_price, "exit_price": f.exit_price,
        "pnl_pts": f.pnl_pts if f.filled else 0.0, "r": r, "mfe_r": mfe_r,
        "mae_r": f.mae_pts / risk_pts if f.filled else 0.0,
        # k-R reached before the trade ended (stop-bar excursions are not counted: pessimistic)
        "hit_1r": f.filled and mfe_r >= 1, "hit_2r": f.filled and mfe_r >= 2,
        "hit_3r": f.filled and mfe_r >= 3,
    }
    if lots is not None:
        row["lots"] = lots
        row["risk_usd"] = lots * risk_pts * lot_value
        row["pnl_usd"] = (f.pnl_pts * lots * lot_value - costs.commission_per_lot * lots) if f.filled else 0.0
    return row


def run_raw(setups: pd.DataFrame, m1: M1, settings: Settings, costs: Costs) -> pd.DataFrame:
    """Every valid setup on its own (no account, no news): the honest edge check."""
    v = setups[setups["status"] == "valid"].sort_values("armed_at")
    rows = []
    for sid, s in v.iterrows():
        armed, od, fd, orr, frr = _deadlines(s, settings, None)
        f = simulate(m1, s["direction"], s["entry"], s["stop"], s["target"], armed, od, fd, costs, orr, frr,
                     order_type=_order_type(s))
        rows.append(_trade_row(sid, s, f, costs, settings.broker.lot_value_per_point, None))
    df = pd.DataFrame(rows).reindex(columns=TRADE_COLUMNS)
    if len(df):
        df["trading_day"] = trading_date(pd.DatetimeIndex(df["armed_at"]), settings.data.trading_day_start).date
    return df


@dataclass
class GuardedResult:
    trades: pd.DataFrame
    blocks: pd.DataFrame
    events: pd.DataFrame                       # halts, resets, firm-limit breaches
    daily: pd.DataFrame                        # per trading day: start/end balance, min equity, pnl
    accounts: int = 1
    notes: list[str] = field(default_factory=list)


def run_guarded(setups: pd.DataFrame, m1: M1, settings: Settings, costs: Costs,
                cal: NewsCalendar) -> GuardedResult:
    lim = Limits.from_settings(settings)
    state = AccountState.fresh(lim)
    v = setups[setups["status"] == "valid"].sort_values("armed_at")
    lot_value = settings.broker.lot_value_per_point
    trades, blocks, events = [], [], []
    pending: list[tuple[pd.Timestamp, float, dict]] = []   # (exit_time, pnl, trade row) not yet applied
    busy_until: pd.Timestamp | None = None
    daily: dict = {}
    account_no = 1
    reset_next_day = False
    logged: set = set()

    def day_rec(td):
        return daily.setdefault(td, {"account": account_no, "start_balance": state.balance,
                                     "end_balance": state.balance, "min_equity": state.balance,
                                     "pnl": 0.0, "trades": 0})

    def apply_closed(upto: pd.Timestamp | None):
        nonlocal pending
        keep = []
        for ex, pnl, row in sorted(pending, key=lambda x: x[0]):
            if upto is None or ex <= upto:
                td = row["trading_day"]
                rec = day_rec(td)
                rec["min_equity"] = min(rec["min_equity"], state.balance - row["_mae_usd"])
                state.record_close(pnl)
                rec["min_equity"] = min(rec["min_equity"], state.balance)
                rec["pnl"] += pnl
                rec["end_balance"] = state.balance
                row["balance_after"] = state.balance
                _check_breach(td, ex)
            else:
                keep.append((ex, pnl, row))
        pending = keep

    def _log_once(td, when, event, detail):
        key = (account_no, td, event)
        if key not in logged:
            logged.add(key)
            events.append({"time": when, "account": account_no, "event": event, "detail": detail})

    def _check_breach(td, when):
        rec = daily[td]
        breached = False
        if rec["start_balance"] - rec["min_equity"] > lim.daily_limit_usd:
            _log_once(td, when, "FIRM_DAILY_LIMIT_BREACH",
                      f"intraday equity {rec['min_equity']:.2f} vs day start {rec['start_balance']:.2f}")
            breached = True
        if rec["min_equity"] < lim.initial_balance - lim.max_dd_usd:
            _log_once(td, when, "FIRM_MAX_DRAWDOWN_BREACH", f"equity {rec['min_equity']:.2f}")
            breached = True
        if breached and not state.halted:
            state.halted = True                     # a failed account takes no more trades
            state.halt_reason = "firm limit breached: account failed"
        if state.halted:
            _log_once(None, when, "HALT", state.halt_reason)

    for sid, s in v.iterrows():
        armed = pd.Timestamp(s["armed_at"])
        apply_closed(armed)
        if state.halted and settings.backtest.reset_on_halt:
            reset_next_day = True
        if busy_until is not None and busy_until <= armed:
            busy_until = None
        state.position_open = busy_until is not None
        td = trading_date(pd.DatetimeIndex([armed]), settings.data.trading_day_start).date[0]
        if td != state.trading_day:
            if reset_next_day and settings.backtest.reset_on_halt:
                account_no += 1
                state = AccountState.fresh(lim)
                events.append({"time": armed, "account": account_no, "event": "RESET",
                               "detail": "new account after halt/breach (backtest only; live needs a manual reset)"})
                reset_next_day = False
            state.roll_day(td)
        day_rec(td)

        sig = SignalCandidate(time=armed, session=s["session"], stop_pts=float(s["risk_pts"]),
                              direction=s["direction"], setup_type=s["setup_type"])
        dec = evaluate(sig, state, cal.check(armed), settings)
        if not dec.allowed:
            blocks.append({"setup_id": sid, "time": armed, "account": account_no,
                           "setup_type": s["setup_type"], "reasons": dec.reason_text})
            continue

        state.record_signal(s["session"])
        armed, od, fd, orr, frr = _deadlines(s, settings, cal)
        f = simulate(m1, s["direction"], s["entry"], s["stop"], s["target"], armed, od, fd, costs, orr, frr,
                     order_type=_order_type(s))
        row = _trade_row(sid, s, f, costs, lot_value, dec.size.lots)
        row["trading_day"] = td
        row["account"] = account_no
        row["_mae_usd"] = f.mae_pts * dec.size.lots * lot_value if f.filled else 0.0
        if f.filled:
            daily[td]["trades"] += 1
            pending.append((f.exit_time, row["pnl_usd"], row))
            busy_until = f.exit_time
        else:
            row["balance_after"] = state.balance
            busy_until = od     # the unfilled order blocks new signals until it is cancelled
        trades.append(row)

    apply_closed(None)
    tdf = pd.DataFrame(trades)
    if len(tdf):
        tdf = tdf.drop(columns=["_mae_usd"]).reindex(columns=TRADE_COLUMNS + ["account"])
    ddf = pd.DataFrame.from_dict(daily, orient="index").rename_axis("trading_day").reset_index()
    return GuardedResult(trades=tdf, blocks=pd.DataFrame(blocks), events=pd.DataFrame(events),
                         daily=ddf, accounts=account_no)


def sizing_preview(setups: pd.DataFrame, settings: Settings) -> pd.Series:
    """How many valid setups would be blocked by lot sizing alone (min lot too big for the risk)."""
    v = setups[setups["status"] == "valid"]
    return v["risk_pts"].map(lambda x: position_size(float(x), settings).blocked_reason or "ok").value_counts()
