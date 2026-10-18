"""Live (paper) engine: streams 1m bars, runs the SAME detection + risk code as the backtest.

Per poll (every live.poll_seconds):
  1. refresh the news calendar when due (fail-safe: stale/failed news -> no signals + alert)
  2. fetch newly completed 1m bars; feed errors or a silent feed in a session -> NO-DATA + alert
  3. track every open paper signal with backtest.fills.simulate on the bars seen so far
  4. on every completed 5m bar inside a session: rebuild the timeframes (stored history before
     today's trading-day start + today's live bars), run setups.detect.detect_all, and pass each
     newly armed valid setup through risk.guard.evaluate. Allowed -> signal (SQLite + alerts),
     blocked -> compliance log with every reason.
Higher-timeframe bars are only included once complete (end <= now), exactly as in the backtest.
Account state for the guard is rebuilt from the database on every decision (paper outcomes by
default, or Rald's journal entries), so a restart never loses or invents state.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.alerts.notify import Notifier, format_signal
from src.backtest.engine import _deadlines, costs_from_settings
from src.backtest.fills import M1, simulate
from src.config import PROJECT_ROOT, Settings, data_dir
from src.data.cleaning import clean_m1
from src.data.dukascopy_live import FeedError
from src.data.providers import DataProvider
from src.data.resample import resample
from src.live.store import Store, iso
from src.news.calendar import BlackoutStatus, LiveNewsCalendar
from src.risk.guard import AccountState, Limits, SignalCandidate, evaluate
from src.setups.detect import detect_all
from src.timeutils import NY, hhmm_minutes, session_of, trading_date

log = logging.getLogger("live")
FIVE = pd.Timedelta(minutes=5)
ONE = pd.Timedelta(minutes=1)


def empty_m1() -> pd.DataFrame:
    """Empty 1m frame with a UTC DatetimeIndex (so time comparisons work before any data arrives)."""
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in ("open", "high", "low", "close", "volume")},
                        index=pd.DatetimeIndex([], tz="UTC", name="ts"))


def trading_day_start_utc(t: pd.Timestamp, day_start: str) -> pd.Timestamp:
    """UTC start (e.g. 18:00 NY the previous evening) of the trading day containing t."""
    td = trading_date(pd.DatetimeIndex([pd.Timestamp(t)]), day_start)[0]
    wall = pd.Timestamp(td) - pd.Timedelta(days=1) + pd.Timedelta(minutes=hhmm_minutes(day_start))
    return wall.tz_localize(NY).tz_convert("UTC")


def td_str(t, settings: Settings) -> str:
    return str(trading_date(pd.DatetimeIndex([pd.Timestamp(t)]), settings.data.trading_day_start).date[0])


class LiveEngine:
    def __init__(self, settings: Settings, provider: DataProvider, news: LiveNewsCalendar, store: Store,
                 notifier: Notifier | None = None,
                 history_loader: Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None):
        settings.require_complete()
        self.s = settings
        self.provider = provider
        self.news = news
        self.store = store
        self.notifier = notifier or Notifier(env={})
        self.limits = Limits.from_settings(settings)
        self.costs = costs_from_settings(settings)
        det = settings.detectors
        self.tf = det.setup_timeframe
        self.need = sorted({self.tf, det.premium_discount.range_timeframe, *det.htf_bias.timeframes})
        self._load_hist = history_loader or self._load_from_store
        self.listeners: list[Callable[[str, dict], None]] = []
        self.hist: dict[str, pd.DataFrame] = {}
        self.m1 = empty_m1()
        self.last_bar: pd.Timestamp | None = None        # start time of the newest 1m bar
        self.last_close_processed: pd.Timestamp | None = None
        self.feed_ok, self.feed_error, self.feed_lag = True, None, None
        self.feed_failures = 0
        self.started_at: pd.Timestamp | None = None
        self.next_news_refresh: pd.Timestamp | None = None
        self.status, self.status_reason = "STARTING", ""
        self.session: str | None = None
        self._alerted: set[str] = set()
        self.last_detections = None
        self.confidence_notes = self._backtest_win_rates()
        from src.live.rules_desk import RulesDesk
        self.rules = RulesDesk(self)

    # --------------------------------------------------------------------------- events
    def emit(self, kind: str, data: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(kind, data)
            except Exception as e:  # noqa: BLE001
                log.warning("listener failed: %s", e)

    def alert(self, now, text: str, level: str = "info", key: str | None = None) -> None:
        if key is not None:
            if key in self._alerted:
                return
            self._alerted.add(key)
        self.store.add_event(now, level, text)
        self.notifier.send(text, level)
        self.emit("alert", {"level": level, "text": text, "time": iso(now)})

    def clear_alert(self, key: str) -> bool:
        if key in self._alerted:
            self._alerted.discard(key)
            return True
        return False

    # -------------------------------------------------------------------------- startup
    def _load_from_store(self, tf: str, start, end) -> pd.DataFrame:
        from src.data.store import load_bars
        try:
            return load_bars(self.s, tf, start, end)
        except FileNotFoundError:
            return pd.DataFrame()

    def bootstrap(self, now: pd.Timestamp) -> None:
        now = pd.Timestamp(now)
        self.started_at = now
        cutoff = trading_day_start_utc(now, self.s.data.trading_day_start)
        for tf in self.need:
            days = self.s.live.lookback_days.get(tf, 60)
            h = self._load_hist(tf, now - pd.Timedelta(days=days), cutoff)
            self.hist[tf] = h[h["end"] <= cutoff] if len(h) and "end" in h else h
        stored = self._load_hist("1m", cutoff, now)
        stored = stored[["open", "high", "low", "close", "volume"]] if len(stored) else stored
        start = stored.index[-1] + ONE if len(stored) else cutoff
        fresh = pd.DataFrame()
        try:
            fresh = self.provider.fetch_history(start, now)
            self.feed_ok, self.feed_error = True, None
        except FeedError as e:
            self.feed_ok, self.feed_error = False, str(e)
        fresh = fresh[fresh.index + ONE <= now] if len(fresh) else fresh
        if len(fresh):
            fresh, _ = clean_m1(fresh, self.s)
        parts = [p for p in (stored, fresh) if len(p)]
        self.m1 = pd.concat(parts).sort_index() if parts else empty_m1()
        self.m1 = self.m1[~self.m1.index.duplicated(keep="last")]
        self.last_bar = self.m1.index[-1] if len(self.m1) else None
        # only the newest completed 5m bar may still produce a signal after a (re)start
        self.last_close_processed = self._floor5(now) - FIVE
        self.refresh_news(now)
        self.alert(now, f"engine started ({self.s.live.mode} mode, {len(self.m1)} live 1m bars since "
                        f"{cutoff:%Y-%m-%d %H:%M} UTC)", "info")
        if not self.feed_ok:
            self.alert(now, f"data feed disconnected: {self.feed_error}", "critical", key="feed")
        self._update_status(now)
        if self.status == "NO-NEWS":
            self.alert(now, f"news data unavailable at startup, no signals: {self.status_reason}", "critical")

    def stop(self, now) -> None:
        self.alert(now, "engine stopped", "warning")

    # ------------------------------------------------------------------------------ news
    def refresh_news(self, now) -> None:
        ok = self.news.refresh(now)
        if not ok and self.news.calendar is None:
            # e.g. HTTP 429 right after a restart: fall back to the last archived snapshot if recent
            from src.news.forexfactory import latest_archive
            snap = latest_archive(self.s) if getattr(self.news.provider, "name", "") == "forexfactory" else None
            if snap and pd.Timestamp(now) - snap[0] <= pd.Timedelta(minutes=self.s.news.stale_after_minutes):
                self.news.load_snapshot(*snap)
        self.next_news_refresh = pd.Timestamp(now) + pd.Timedelta(minutes=self.s.news.refresh_minutes)
        if not ok:
            # retry sooner while failing; status() goes fail-safe once data is stale
            self.next_news_refresh = pd.Timestamp(now) + pd.Timedelta(minutes=2)

    # --------------------------------------------------------------------------- polling
    @staticmethod
    def _floor5(t: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(t).floor("5min")

    def poll(self, now: pd.Timestamp) -> None:
        now = pd.Timestamp(now)
        if self.next_news_refresh is None or now >= self.next_news_refresh:
            self.refresh_news(now)
        new = self._fetch_new(now)
        if len(new):
            self._append(new)
            self.emit("bar", {"tf": "1m", "bars": [_bar_json(t, r) for t, r in new.iterrows()]})
        self._heartbeat(now)
        self.update_paper(now)
        self.rules.poll(now)
        if self.last_bar is not None:
            latest_close = self._floor5(self.last_bar + ONE)       # newest 5m boundary we have data up to
            t = self.last_close_processed + FIVE
            while t <= latest_close:
                self.on_bar_close(t, now)
                self.last_close_processed = t
                t += FIVE
        self._update_status(now)
        self._weekly_report(now)

    def _weekly_report(self, now) -> None:
        """Phase 7: every Friday after the NY close, compare paper results with the backtest."""
        ny = pd.Timestamp(now).tz_convert(NY)
        if ny.weekday() != 4 or ny.hour * 60 + ny.minute < 16 * 60 + 15:
            return
        week = ny.strftime("%G-W%V")
        if self.store.get("weekly_report_week") == week:
            return
        self.store.set("weekly_report_week", week, now)
        try:
            from src.live.paper_report import compare, write_html
            res = compare(self.store, self.s)
            out = PROJECT_ROOT / "reports"
            out.mkdir(exist_ok=True)
            write_html(res, out / "paper_latest.html")
            wk = [w for w in res["weeks"] if w["week"] == week]
            this = wk[0] if wk else {"signals": 0, "filled": 0, "total_r": 0.0}
            self.alert(now, f"weekly paper report {week}: {this['signals']} signals, {this['filled']} closed, "
                            f"{this['total_r']:+.2f}R this week. Overall: {res['verdict']}",
                       "warning" if res["drift"] else "info")
        except (OSError, ValueError, KeyError) as e:
            self.alert(now, f"weekly paper report failed: {e}", "warning")

    def _fetch_new(self, now) -> pd.DataFrame:
        since = (self.last_bar + ONE) if self.last_bar is not None else now - pd.Timedelta(hours=6)
        need = int((now - since) / ONE) + 5
        try:
            df = self.provider.fetch_before(now, max(need, 5))
        except FeedError as e:
            # status goes NO-DATA at once (no signals); the alert waits for a 2nd failure in a row,
            # because single handshake timeouts are common on this feed and recover within seconds
            self.feed_failures += 1
            if self.feed_failures >= 2:
                self.alert(now, f"data feed disconnected: {e}", "critical", key="feed")
            self.feed_ok, self.feed_error = False, str(e)
            return pd.DataFrame()
        self.feed_failures = 0
        if not self.feed_ok and self.clear_alert("feed"):
            self.alert(now, "data feed reconnected", "info")
        self.feed_ok, self.feed_error = True, None
        df = df[(df.index >= since) & (df.index + ONE <= now)]
        if df.empty:
            return df
        ctx = self.m1.tail(300)
        cleaned, _ = clean_m1(pd.concat([ctx, df]) if len(ctx) else df, self.s)
        return cleaned[cleaned.index >= since]

    def _append(self, new: pd.DataFrame) -> None:
        self.m1 = pd.concat([self.m1, new[["open", "high", "low", "close", "volume"]]]).sort_index()
        self.m1 = self.m1[~self.m1.index.duplicated(keep="last")]
        self.last_bar = self.m1.index[-1]

    def _heartbeat(self, now) -> None:
        if self.last_bar is None:
            return
        self.feed_lag = (now - (self.last_bar + ONE)).total_seconds()
        in_sess = session_of(pd.DatetimeIndex([now]), self.s.sessions).iloc[0] is not None
        limit = self.s.live.heartbeat_seconds + 30
        if in_sess and self.feed_lag > limit:
            self.alert(now, f"no new US100 data for {self.feed_lag:.0f} s during the session", "critical",
                       key="heartbeat")
        elif self.feed_lag <= limit and self.clear_alert("heartbeat"):
            self.alert(now, "US100 data flowing again", "info")

    # ------------------------------------------------------------------------ timeframes
    def frames(self, upto: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """Every timeframe using only bars complete at `upto` (history + live)."""
        m1 = self.m1[self.m1.index < upto]
        out = {}
        for tf in self.need:
            live = resample(m1, tf, self.s.data.trading_day_start, self.s.data.market_close) if len(m1) else None
            parts = [p for p in (self.hist.get(tf), live) if p is not None and len(p)]
            df = pd.concat(parts).sort_index() if parts else pd.DataFrame()
            if len(df):
                df = df[~df.index.duplicated(keep="last")]
                df = df[df["end"] <= upto]
            out[tf] = df
        return out

    # ------------------------------------------------------------------------- detection
    def on_bar_close(self, t: pd.Timestamp, wall: pd.Timestamp | None = None) -> None:
        """A 5m bar ending at t just completed (`wall` = real time we are processing it)."""
        self._wall = wall if wall is not None else t
        bar_start = t - FIVE
        if session_of(pd.DatetimeIndex([bar_start]), self.s.sessions).iloc[0] is None:
            return
        fr = self.frames(t)
        if fr[self.tf].empty:
            return
        det = detect_all(fr[self.tf], {k: v for k, v in fr.items() if k != self.tf}, self.s)
        self.last_detections = det
        st = det.setups
        if st.empty:
            return
        new = st[(st["status"] == "valid") & (st["armed_at"] == t)]
        for _, s in new.iterrows():
            self.consider(s, t)

    def setup_key(self, s) -> str:
        return f"{iso(s['armed_at'])}|{s['setup_type']}|{s['direction']}|{s['entry']:.3f}"

    def consider(self, s, now: pd.Timestamp) -> None:
        key = self.setup_key(s)
        if self.store.has_setup(key):
            return
        state = self.account_state(now)
        news = self.news.status(now)
        sig = SignalCandidate(time=now, session=s["session"], stop_pts=float(s["risk_pts"]),
                              direction=s["direction"], setup_type=s["setup_type"])
        dec = evaluate(sig, state, news, self.s)
        late = (getattr(self, "_wall", now) - now).total_seconds()
        if late > self.s.live.max_signal_delay_seconds:
            # fail safe: data arrived late (feed outage), the entry may already be gone
            dec = type(dec)(allowed=False, reasons=(*dec.reasons, f"late_data_{late:.0f}s"), size=dec.size)
        td = td_str(now, self.s)
        if not dec.allowed:
            self.store.add_block(key, now, td, s["setup_type"], s["direction"], list(dec.reasons))
            self.emit("block", {"time": iso(now), "setup_type": s["setup_type"], "direction": s["direction"],
                                "reasons": list(dec.reasons)})
            return
        armed, od, fd, orr, frr = _deadlines(s, self.s, self.news.calendar)
        row = {
            "setup_key": key, "created_at": iso(now), "trading_day": td,
            "setup_type": s["setup_type"], "direction": s["direction"], "session": s["session"],
            "entry": float(s["entry"]), "stop": float(s["stop"]), "target": float(s["target"]),
            "rr": float(s["rr"]), "risk_pts": float(s["risk_pts"]), "lots": dec.size.lots,
            "risk_usd": dec.size.risk_usd, "expires_at": iso(s["expires_at"]), "order_deadline": iso(od),
            "flat_deadline": iso(fd), "order_reason": orr, "flat_reason": frr,
            "reason": reason_text(s, self.s), "confidence": None,
            "confidence_note": self.confidence_notes.get(s["setup_type"], "no model (Phase 4 skipped)"),
            "status": "active",
        }
        sid = self.store.add_signal(row)
        full = self.store.signal(sid)
        self.alert(now, format_signal(full, self.price_offset()), "signal")
        self.emit("signal", full)

    # ----------------------------------------------------------------------- paper fills
    def update_paper(self, now: pd.Timestamp) -> None:
        open_sigs = self.store.open_signals()
        if not open_sigs or self.m1.empty:
            return
        m1 = M1.from_frame(self.m1)
        lot_value = self.s.broker.lot_value_per_point
        changed = False
        from src.live.rules_desk import RULES
        for sg in open_sigs:
            if sg["setup_type"] in RULES:
                continue                       # tracked by the rules desk with its own (backtest) code
            armed = pd.Timestamp(sg["created_at"])
            od, fd = pd.Timestamp(sg["order_deadline"]), pd.Timestamp(sg["flat_deadline"])
            f = simulate(m1, sg["direction"], sg["entry"], sg["stop"], sg["target"], armed, od, fd,
                         self.costs, sg["order_reason"], sg["flat_reason"])
            final = f.outcome in ("tp", "sl") or (not f.filled and now >= od) or (f.filled and now >= fd)
            if not final:
                status = "filled" if f.filled else "active"
                if status != sg["status"]:
                    self.store.update_signal(sg["id"], status=status, paper_fill_time=iso(f.fill_time))
                    self.emit("signal", self.store.signal(sg["id"]))
                    if status == "filled":
                        self.alert(now, f"paper: signal #{sg['id']} filled at {f.fill_price:.1f}", "info")
                continue
            if f.filled:
                r = f.pnl_pts / sg["risk_pts"] - (self.costs.commission_per_lot / (sg["risk_pts"] * lot_value))
                pnl = f.pnl_pts * sg["lots"] * lot_value - self.costs.commission_per_lot * sg["lots"]
                self.store.update_signal(sg["id"], status="closed", paper_outcome=f.outcome,
                                         paper_fill_time=iso(f.fill_time), paper_exit_time=iso(f.exit_time),
                                         paper_exit_price=f.exit_price, paper_r=r, paper_pnl_usd=pnl)
                self.alert(now, f"paper: signal #{sg['id']} closed ({f.outcome}) {r:+.2f}R ${pnl:+.2f}", "info")
            else:
                self.store.update_signal(sg["id"], status="cancelled" if f.outcome == "news_cancel" else "expired",
                                         paper_outcome=f.outcome)
            self.emit("signal", self.store.signal(sg["id"]))
            changed = True
        if changed:
            self._check_halt(now)

    def _check_halt(self, now) -> None:
        st = self.account_state(now)
        if st.drawdown >= self.limits.dd_buffer_usd and not self.store.get("halted", False):
            self.store.set("halted", True, now)
            self.alert(now, f"HALTED: drawdown ${st.drawdown:.2f} reached the internal buffer "
                            f"${self.limits.dd_buffer_usd:.2f}. Manual reset required.", "critical")

    # -------------------------------------------------------------------- account state
    def closed_trades(self) -> pd.DataFrame:
        """Closed trades from the configured source with exit time and $ result."""
        if self.s.live.state_source == "journal":
            rows = self.store._all("SELECT action_at AS exit_time, pnl_usd FROM signals "
                                   "WHERE action='taken' AND pnl_usd IS NOT NULL")
        else:
            rows = self.store._all("SELECT paper_exit_time AS exit_time, paper_pnl_usd AS pnl_usd FROM signals "
                                   "WHERE paper_exit_time IS NOT NULL AND paper_pnl_usd IS NOT NULL")
        df = pd.DataFrame(rows, columns=["exit_time", "pnl_usd"])
        if len(df):
            df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
            df = df.sort_values("exit_time")
        return df

    def account_state(self, now: pd.Timestamp) -> AccountState:
        st = AccountState.fresh(self.limits)
        tr = self.closed_trades()
        tr = tr[tr["exit_time"] <= now] if len(tr) else tr
        override = self.store.get("balance_override")
        if override:
            t0 = pd.Timestamp(override["time"])
            balance = float(override["value"]) + (tr.loc[tr["exit_time"] > t0, "pnl_usd"].sum() if len(tr) else 0.0)
        else:
            balance = self.limits.initial_balance + (tr["pnl_usd"].sum() if len(tr) else 0.0)
        td = td_str(now, self.s)
        if len(tr):
            tdays = trading_date(pd.DatetimeIndex(tr["exit_time"]), self.s.data.trading_day_start).date
            today = tr[np.array([str(d) == td for d in tdays])]
        else:
            today = tr
        st.balance = balance
        st.trading_day = pd.Timestamp(td).date()
        st.day_start_balance = balance - (today["pnl_usd"].sum() if len(today) else 0.0)
        losses = 0
        for p in (today["pnl_usd"] if len(today) else []):
            losses = losses + 1 if p < 0 else (0 if p > 0 else losses)
        st.consecutive_losses = losses
        sigs = self.store._all("SELECT session FROM signals WHERE trading_day=?", (td,))
        st.signals_today = len(sigs)
        for r in sigs:
            st.signals_by_session[r["session"]] = st.signals_by_session.get(r["session"], 0) + 1
        st.position_open = bool(self.store.open_signals())
        st.halted = bool(self.store.get("halted", False))
        st.halt_reason = "max drawdown buffer reached; manual reset required" if st.halted else ""
        return st

    # --------------------------------------------------------------------------- status
    def price_offset(self) -> float:
        return float(self.store.get("price_offset", self.s.broker.price_offset))

    def news_status(self, now) -> BlackoutStatus:
        return self.news.status(now)

    def _update_status(self, now) -> None:
        st = self.account_state(now)
        news = self.news.status(now)
        sess = session_of(pd.DatetimeIndex([now]), self.s.sessions).iloc[0]
        lim = self.limits
        stale = self.feed_lag is not None and sess is not None and self.feed_lag > self.s.live.heartbeat_seconds + 30
        if not self.feed_ok or self.last_bar is None or stale:
            status, why = "NO-DATA", self.feed_error or "no recent bars"
        elif not news.available:
            status, why = "NO-NEWS", news.reason
        elif st.halted:
            status, why = "HALTED", st.halt_reason
        elif st.day_loss >= lim.daily_buffer_usd:
            status, why = "STOPPED-FOR-DAY", f"daily loss ${st.day_loss:.2f} >= buffer ${lim.daily_buffer_usd:.2f}"
        elif st.consecutive_losses >= lim.max_consecutive_losses:
            status, why = "STOPPED-FOR-DAY", f"{st.consecutive_losses} consecutive losses"
        elif st.signals_today >= lim.max_per_day:
            status, why = "STOPPED-FOR-DAY", "max signals for the day reached"
        elif news.blackout or (news.minutes_to_next is not None
                               and news.minutes_to_next <= self.s.news.blackout_minutes_before + 1):
            status, why = "PAUSED-NEWS", news.reason or f"{news.next_event_title} in {news.minutes_to_next:.0f} min"
        elif sess is None:
            status, why = "OUTSIDE-SESSION", ""
        else:
            status, why = "ACTIVE", ""
        prev = self.status
        self.status, self.status_reason, self.session = status, why, sess
        if prev != status and prev != "STARTING":
            if status == "PAUSED-NEWS":
                self.alert(now, f"news blackout starting: {why}", "warning")
            elif prev == "PAUSED-NEWS":
                self.alert(now, "news blackout ended", "info")
            if status == "STOPPED-FOR-DAY":
                self.alert(now, f"daily stop reached: {why}", "warning")
            if status == "NO-NEWS":
                self.alert(now, f"news data unavailable, no signals: {why}", "critical")
            self.emit("status", {"status": status, "status_reason": why})

    def _backtest_win_rates(self) -> dict[str, str]:
        p = PROJECT_ROOT / "reports" / "backtest" / "raw_trades.csv"
        try:
            t = pd.read_csv(p)
            f = t[t["filled"]]
            return {k: f"no model (Phase 4 skipped); backtest {k} win rate {g['r'].gt(0).mean():.0%} "
                       f"over {len(g)} trades, expectancy {g['r'].mean():+.2f}R"
                    for k, g in f.groupby("setup_type")}
        except (OSError, KeyError, ValueError):
            return {}


def reason_text(s, settings: Settings) -> str:
    parts = [f"{s['levels_swept']} swept {s['depth_atr']:.2f}xATR", f"MSS body {s['disp_body_atr']:.1f}xATR"]
    if s["setup_type"] == "sweep_mss_fvg":
        parts.append(f"FVG {settings.setups.entry_mode} entry")
    else:
        parts.append(f"OB {settings.setups.ob_entry} entry")
    if isinstance(s.get("bias"), str):
        parts.append(f"HTF bias {s['bias']}")
    if pd.notna(s.get("pd_pos")):
        parts.append(f"P/D {s['pd_pos']:.2f}")
    parts.append(f"target {s['target_kind']}")
    return ", ".join(parts)


def _bar_json(t, r) -> dict:
    return {"t": iso(t), "o": float(r["open"]), "h": float(r["high"]), "l": float(r["low"]), "c": float(r["close"])}


def default_db_path(settings: Settings) -> Path:
    p = Path(settings.live.db_path)
    if p.is_absolute():
        return p
    root = data_dir(settings)
    rel = p.relative_to(settings.data.root) if p.parts and p.parts[0] == settings.data.root else p
    return root / rel
