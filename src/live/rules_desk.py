"""Live paper desk for the rule-based US100 setups (Rald's choice, 2026-10-18: paper-trade the top candidates).

  crabel_nr4    : at 09:30, if yesterday was an NR4 day, a BRACKET: buy stop at open + stretch, sell stop at
                  open - stretch (stretch = 10-day mean of min(open-low, high-open)). The first to trigger is the
                  trade; the stop is the other level; target 3R; flat 16:00. (Round 9's most stable variant.)
  opening_drive : at 10:00, trade the 09:30-10:00 move direction if |move| >= 0.5 x stretch; market entry,
                  stop at the opposite 09:30-10:00 extreme, target 3R, flat 16:00. (Round 14a.)

The desk calls the SAME functions as the backtest (strategies.variants.crabel / opening_drive) on the partial
day seen so far, so live and backtest cannot drift apart. Signals pass the same risk guard and are stored and
alerted like every other signal; the paper outcome is recomputed from the bars on every poll.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.live.store import iso
from src.risk.guard import SignalCandidate, evaluate
from src.strategies.published import Cost, Day
from src.strategies.variants import crabel, opening_drive
from src.timeutils import NY

RULES = ("crabel_nr4", "opening_drive")
ONE = pd.Timedelta(minutes=1)


class RulesDesk:
    def __init__(self, engine):
        self.e = engine
        s = engine.s
        self.cost = Cost(s.broker.typical_spread_points, s.backtest.stop_slippage_points)

    # --------------------------------------------------------------------------- inputs
    def _daily(self, now) -> pd.DataFrame:
        d1 = self.e.frames(now).get("1d", pd.DataFrame())
        if d1.empty:
            return d1
        d1 = d1.sort_index().copy()
        d1["ny_date"] = pd.DatetimeIndex(d1["end"]).tz_convert(NY).normalize().tz_localize(None)
        return d1

    def _today(self, now) -> Day | None:
        m1 = self.e.m1
        if m1.empty:
            return None
        m1 = m1[m1.index + ONE <= now]
        ny = m1.index.tz_convert(NY)
        today = pd.Timestamp(now).tz_convert(NY).normalize().tz_localize(None)
        mins = ny.hour * 60 + ny.minute
        sel = (ny.normalize().tz_localize(None) == today) & (mins >= 570) & (mins <= 959)
        g = m1[sel]
        if g.empty or int(mins[sel][0]) != 570:
            return None
        return Day(today, g["open"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy(),
                   g["volume"].to_numpy(dtype=float), np.asarray(mins[sel]),
                   pd.DatetimeIndex(g.index).as_unit("ns").asi8, None, None)

    @staticmethod
    def _stretch(d1: pd.DataFrame, today: pd.Timestamp) -> tuple[float | None, bool]:
        hist = d1[d1["ny_date"] < today].tail(10)
        if len(hist) < 10:
            return None, False
        rng = (hist["high"] - hist["low"]).to_numpy()
        stretch = float(np.minimum(hist["open"] - hist["low"], hist["high"] - hist["open"]).mean())
        return (stretch if stretch > 0 else None), bool(rng[-1] < rng[-4:-1].min())

    # ----------------------------------------------------------------------------- poll
    def poll(self, now: pd.Timestamp) -> None:
        now = pd.Timestamp(now)
        day = self._today(now)
        if day is None:
            return
        d1 = self._daily(now)
        if d1.empty:
            return
        stretch, nr4 = self._stretch(d1, day.date)
        if stretch is None:
            return
        td = str(day.date.date())
        op = float(day.o[0])
        # 1) issue signals (once per day each)
        if nr4:
            self._issue(now, td, "crabel_nr4", "bracket", entry=op + stretch, stop=op - stretch, target=None,
                        risk_pts=2 * stretch, reason=f"NR4 yesterday; stretch {stretch:.1f} pts; bracket around "
                                                     f"the 09:30 open {op:.1f}",
                        notes={"long_level": op + stretch, "short_level": op - stretch, "stretch": stretch})
        k10 = day.idx(600)
        if k10 is not None:
            drive = opening_drive(day, self.cost, d1)
            if drive:
                t = drive[0]
                self._issue(now, td, "opening_drive", t["side"], entry=t["entry"], stop=t["stop0"],
                            target=t["entry"] + 3 * (t["entry"] - t["stop0"]), risk_pts=t["risk_pts"],
                            reason=f"09:30-10:00 move {day.c[k10 - 1] - op:+.1f} pts (>= 0.5 x stretch "
                                   f"{stretch:.1f})", notes={})
        # 2) track paper outcomes of today's rule signals
        self._track(now, day, d1, td)

    def _issue(self, now, td, setup, direction, entry, stop, target, risk_pts, reason, notes) -> None:
        key = f"{td}|{setup}"
        if self.e.store.has_setup(key):
            return
        session = "ny_am"
        state = self.e.account_state(now)
        news = self.e.news.status(now)
        sig = SignalCandidate(time=now, session=session, stop_pts=float(risk_pts),
                              direction=direction if direction != "bracket" else "long", setup_type=setup)
        dec = evaluate(sig, state, news, self.e.s)
        # fail safe: a bracket meant for 09:30 or a drive meant for 10:00 that we only see late (feed outage)
        # is not issued; the market may already have moved through the levels
        due = pd.Timestamp(f"{td} {'09:31' if setup == 'crabel_nr4' else '10:01'}", tz=NY).tz_convert("UTC")
        late = (now - due).total_seconds()
        if late > self.e.s.live.max_signal_delay_seconds:
            dec = type(dec)(allowed=False, reasons=(*dec.reasons, f"late_data_{late:.0f}s"), size=dec.size)
        if not dec.allowed:
            self.e.store.add_block(key, now, td, setup, direction, list(dec.reasons))
            self.e.emit("block", {"time": iso(now), "setup_type": setup, "direction": direction,
                                  "reasons": list(dec.reasons)})
            return
        end = pd.Timestamp(f"{td} 16:00", tz=NY).tz_convert("UTC")
        row = {"setup_key": key, "created_at": iso(now), "trading_day": td, "setup_type": setup,
               "direction": direction, "session": session, "entry": float(entry), "stop": float(stop),
               "target": None if target is None else float(target), "rr": 3.0, "risk_pts": float(risk_pts),
               "lots": dec.size.lots, "risk_usd": dec.size.risk_usd, "expires_at": iso(end),
               "order_deadline": iso(end), "flat_deadline": iso(end), "order_reason": "eod", "flat_reason": "eod",
               "reason": reason, "confidence": None,
               "confidence_note": "rule setup (no model); paper only: being validated on live data",
               "status": "active", "notes": json.dumps(notes)}
        sid = self.e.store.add_signal(row)
        full = self.e.store.signal(sid)
        if direction == "bracket":
            text = (f"PAPER: US100 Crabel NR4 BRACKET: BUY STOP {notes['long_level']:.1f} / SELL STOP "
                    f"{notes['short_level']:.1f}. When one fills cancel the other; stop = the other level; "
                    f"target 3R; flat by 16:00 NY. {dec.size.lots:.2f} lots (risk ${dec.size.risk_usd:.2f}).")
        else:
            text = (f"PAPER: US100 opening drive {direction.upper()} at market ~{entry:.1f}, stop {stop:.1f}, "
                    f"target {target:.1f} (3R), flat by 16:00 NY. {dec.size.lots:.2f} lots.")
        self.e.alert(now, text, "signal")
        self.e.emit("signal", full)

    def _track(self, now, day: Day, d1: pd.DataFrame, td: str) -> None:
        end = pd.Timestamp(f"{td} 16:00", tz=NY).tz_convert("UTC")
        lot_value = self.e.s.broker.lot_value_per_point
        for sg in self.e.store.open_signals():
            if sg["setup_type"] not in RULES or sg["trading_day"] != td:
                continue
            out = crabel(day, self.cost, d1, "NR4", 1.0, 3.0) if sg["setup_type"] == "crabel_nr4" else \
                opening_drive(day, self.cost, d1)
            if not out:
                if now >= end:
                    self.e.store.update_signal(sg["id"], status="expired", paper_outcome="not_triggered")
                    self.e.emit("signal", self.e.store.signal(sg["id"]))
                continue
            t = out[0]
            final = t["reason"] in ("sl", "tp") or now >= end
            fields = {"direction": t["side"], "paper_fill_time": iso(t["entry_time"])}
            if sg["setup_type"] == "crabel_nr4":
                fields.update(entry=t["entry"], stop=t["stop0"], target=t["entry"] + 3 * (t["entry"] - t["stop0"]))
            if final:
                pnl = t["pnl_pts"] * sg["lots"] * lot_value
                fields.update(status="closed", paper_outcome=t["reason"], paper_exit_time=iso(t["exit_time"]),
                              paper_exit_price=t["exit"], paper_r=t["r"], paper_pnl_usd=pnl)
                self.e.alert(now, f"paper: {sg['setup_type']} #{sg['id']} closed ({t['reason']}) {t['r']:+.2f}R "
                                  f"${pnl:+.2f}", "info")
            elif sg["status"] != "filled":
                fields["status"] = "filled"
                self.e.alert(now, f"paper: {sg['setup_type']} #{sg['id']} filled {t['side']} at {t['entry']:.1f}",
                             "info")
            self.e.store.update_signal(sg["id"], **fields)
            self.e.emit("signal", self.e.store.signal(sg["id"]))
