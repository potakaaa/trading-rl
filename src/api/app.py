"""FastAPI backend: REST + WebSocket for the dashboard, with the live engine running in-process.

Start everything with one command:  uv run python -m src.cli live run
Contract: docs/API.md.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import PROJECT_ROOT, Settings
from src.live.engine import LiveEngine, td_str
from src.live.store import iso
from src.timeutils import trading_date

log = logging.getLogger("api")


class Hub:
    """Fan-out of engine events to connected WebSocket clients (thread-safe enqueue)."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, kind: str, data) -> None:
        if self.loop is None or not self.clients:
            return
        msg = json.dumps({"type": kind, "data": data}, default=str)
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, msg), self.loop)

    async def _send(self, ws: WebSocket, msg: str) -> None:
        try:
            await ws.send_text(msg)
        except Exception:  # noqa: BLE001
            self.clients.discard(ws)


class ActionBody(BaseModel):
    action: str
    fill_price: float | None = None
    exit_price: float | None = None
    lots: float | None = None
    notes: str | None = None


class BalanceBody(BaseModel):
    balance: float


class OffsetBody(BaseModel):
    offset: float


def signal_json(r: dict, offset: float) -> dict:
    out = {k: r.get(k) for k in ("id", "created_at", "setup_type", "direction", "session", "entry", "stop",
                                 "target", "rr", "risk_pts", "lots", "risk_usd", "expires_at", "confidence",
                                 "confidence_note", "reason", "status", "action", "fill_price", "exit_price",
                                 "pnl_usd")}
    out["notes"] = r.get("notes") or ""
    for k in ("entry", "stop", "target"):
        out[f"{k}_broker"] = None if r.get(k) is None else r[k] + offset
    out["paper"] = {"outcome": r.get("paper_outcome"), "fill_time": r.get("paper_fill_time"),
                    "exit_time": r.get("paper_exit_time"), "r": r.get("paper_r")}
    return out


def create_app(settings: Settings, engine: LiveEngine, run_engine: bool = True) -> FastAPI:
    hub = Hub()
    engine.listeners.append(lambda kind, data: hub.publish(kind, _ws_payload(kind, data, engine)))
    engine.notifier.sink = None   # alerts already reach the hub via engine.emit("alert")

    async def loop_task():
        now = pd.Timestamp.now(tz="UTC")
        await asyncio.to_thread(engine.bootstrap, now)
        tick = 0
        while True:
            await asyncio.sleep(settings.live.poll_seconds)
            try:
                await asyncio.to_thread(engine.poll, pd.Timestamp.now(tz="UTC"))
            except Exception as e:  # noqa: BLE001 - fail safe: report, keep the server alive
                log.exception("engine poll failed")
                engine.status, engine.status_reason = "NO-DATA", f"engine error: {e}"
                engine.alert(pd.Timestamp.now(tz="UTC"), f"engine error, no signals: {e}", "critical",
                             key="engine_error")
            tick += 1
            hub.publish("status", status_payload(engine))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        task = asyncio.create_task(loop_task()) if run_engine else None
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            engine.stop(pd.Timestamp.now(tz="UTC"))

    app = FastAPI(title="US100 signal assistant", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                       allow_methods=["*"], allow_headers=["*"])
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    app.mount("/reports", StaticFiles(directory=reports), name="reports")

    @app.get("/api/status")
    def status():
        return status_payload(engine)

    @app.get("/api/bars")
    def bars(tf: str = "5m", limit: int = 500):
        if tf not in ("1m", "5m", "15m"):
            raise HTTPException(400, "tf must be 1m, 5m or 15m")
        now = pd.Timestamp.now(tz="UTC")
        fr = engine.frames(now) if tf != "1m" else None
        df = engine.m1 if tf == "1m" else _with_hist(engine, tf, fr)
        df = df.tail(min(limit, 5000))
        return {"tf": tf, "bars": [{"t": iso(t), "o": r.open, "h": r.high, "l": r.low, "c": r.close}
                                   for t, r in df.iterrows()]}

    @app.get("/api/overlays")
    def overlays(day: str | None = None):
        return overlays_payload(engine, day)

    @app.get("/api/signals")
    def signals(limit: int = 50):
        off = engine.price_offset()
        return {"signals": [signal_json(r, off) for r in engine.store.signals(limit)]}

    @app.post("/api/signals/{sid}/action")
    def action(sid: int, body: ActionBody):
        r = engine.store.signal(sid)
        if r is None:
            raise HTTPException(404, "signal not found")
        if body.action not in ("taken", "skipped"):
            raise HTTPException(400, "action must be taken or skipped")
        fields = {"action": body.action, "action_at": iso(pd.Timestamp.now(tz="UTC"))}
        for k, col in (("fill_price", "fill_price"), ("exit_price", "exit_price"), ("lots", "lots_taken")):
            v = getattr(body, k)
            if v is not None:
                fields[col] = v
        if body.notes is not None:
            fields["notes"] = body.notes
        r.update(fields)
        fill, exit_, lots = r.get("fill_price"), r.get("exit_price"), r.get("lots_taken") or r.get("lots")
        if body.action == "taken" and fill is not None and exit_ is not None and lots:
            # journal prices are BROKER prices; P&L does not depend on the offset
            pts = (fill - exit_) if r["direction"] == "short" else (exit_ - fill)
            fields["pnl_usd"] = pts * lots * settings.broker.lot_value_per_point \
                - settings.backtest.commission_per_lot * lots
        engine.store.update_signal(sid, **fields)
        out = signal_json(engine.store.signal(sid), engine.price_offset())
        hub.publish("signal", out)
        return out

    @app.get("/api/blocks")
    def blocks(day: str | None = None):
        d = day or td_str(pd.Timestamp.now(tz="UTC"), settings)
        return {"blocks": [{"time": b["time"], "setup_type": b["setup_type"], "direction": b["direction"],
                            "reasons": b["reasons"]} for b in engine.store.blocks(d)]}

    @app.get("/api/news")
    def news():
        cal = engine.news.calendar
        events = []
        if cal is not None:
            for _, e in cal.events.iterrows():
                events.append({"ts": iso(e["ts"]), "title": e["title"],
                               "blackout_start": iso(e["ts"] - cal.before), "blackout_end": iso(e["ts"] + cal.after)})
        return {"available": engine.news.status().available, "last_refresh": iso(engine.news.last_ok),
                "events": events}

    @app.get("/api/journal")
    def journal(start: str | None = None, end: str | None = None, setup_type: str | None = None,
                action: str | None = None):
        where, args = ["1=1"], []
        if start:
            where.append("trading_day >= ?")
            args.append(start)
        if end:
            where.append("trading_day <= ?")
            args.append(end)
        if setup_type:
            where.append("setup_type = ?")
            args.append(setup_type)
        if action:
            where.append("action = ?")
            args.append(action)
        rows = engine.store.signals(10000, "WHERE " + " AND ".join(where), tuple(args))
        taken = [r for r in rows if r["action"] == "taken" and r["pnl_usd"] is not None]
        paper = [r for r in rows if r["paper_r"] is not None and r["paper_exit_time"]]
        risk = settings.account.initial_balance * settings.account.risk_per_trade_pct / 100
        stats = {"taken": sum(r["action"] == "taken" for r in rows),
                 "wins": sum(r["pnl_usd"] > 0 for r in taken), "losses": sum(r["pnl_usd"] < 0 for r in taken),
                 "total_r": sum(r["pnl_usd"] for r in taken) / risk if taken else 0.0,
                 "total_pnl_usd": sum(r["pnl_usd"] for r in taken),
                 "paper_total_r": sum(r["paper_r"] for r in paper), "paper_trades": len(paper)}
        off = engine.price_offset()
        return {"signals": [signal_json(r, off) for r in rows], "stats": stats}

    @app.get("/api/backtest")
    def backtest():
        summary = {}
        p = reports / "backtest" / "raw_trades.csv"
        if p.exists():
            from src.backtest import metrics
            t = pd.read_csv(p)
            summary["raw"] = _clean(metrics.summary(t))
        g = reports / "backtest" / "guarded_trades.csv"
        if g.exists() and g.stat().st_size > 5:
            from src.backtest import metrics
            summary["guarded"] = _clean(metrics.summary(pd.read_csv(g)))
        gate = None
        if "raw" in summary:
            e = summary["raw"]["expectancy_r"]
            gate = {"pass": bool(e is not None and e > 0),
                    "text": f"raw expectancy {e:+.3f} R over {summary['raw']['filled']} trades"
                    if e is not None else "no trades"}
        def url(name):
            return f"/reports/{name}" if (reports / name).exists() else None

        return {"report_url": url("backtest_latest.html"), "research_url": url("research_latest.html"),
                "summary": summary, "gate": gate, "model_version": None}

    @app.get("/api/settings")
    def get_settings():
        return json.loads(settings.model_dump_json())

    @app.post("/api/account/balance")
    def set_balance(body: BalanceBody):
        engine.store.set("balance_override", {"value": body.balance, "time": iso(pd.Timestamp.now(tz="UTC"))})
        return status_payload(engine)

    @app.post("/api/account/reset")
    def reset():
        now = pd.Timestamp.now(tz="UTC")
        engine.store.set("halted", False, now)
        engine.alert(now, "halt manually reset by Rald", "warning")
        return status_payload(engine)

    @app.post("/api/price-offset")
    def set_offset(body: OffsetBody):
        engine.store.set("price_offset", body.offset)
        return status_payload(engine)

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        hub.clients.add(socket)
        try:
            await socket.send_text(json.dumps({"type": "status", "data": status_payload(engine)}, default=str))
            while True:
                await socket.receive_text()     # keepalive / ignore client messages
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(socket)

    return app


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        try:
            fv = float(v)
            out[k] = None if pd.isna(fv) else fv
        except (TypeError, ValueError):
            out[k] = v
    return out


def _with_hist(engine: LiveEngine, tf: str, fr: dict | None) -> pd.DataFrame:
    if fr is not None and tf in fr and len(fr[tf]):
        return fr[tf]
    from src.data.resample import resample
    return resample(engine.m1, tf, engine.s.data.trading_day_start, engine.s.data.market_close) \
        if len(engine.m1) else pd.DataFrame(columns=["open", "high", "low", "close"])


def _ws_payload(kind: str, data, engine: LiveEngine):
    if kind == "signal" and isinstance(data, dict) and "setup_key" in data:
        return signal_json(data, engine.price_offset())
    if kind == "status":
        return status_payload(engine)
    if kind == "bar" and isinstance(data, dict) and "bars" in data:
        bars = data["bars"]
        return {"tf": "1m", "bar": bars[-1], "bars": bars}
    return data


def status_payload(engine: LiveEngine) -> dict:
    s = engine.s
    now = pd.Timestamp.now(tz="UTC")
    st = engine.account_state(now)
    lim = engine.limits
    news = engine.news.status(now)
    price = float(engine.m1["close"].iloc[-1]) if len(engine.m1) else None
    warnings = [f"{k} is UNCONFIRMED ({getattr(s.broker, k)})" for k in s.broker.unconfirmed]
    warnings.append(f"news window {s.news.blackout_minutes_before}/{s.news.blackout_minutes_after} min "
                    "(stricter of the published values; confirm with Equity Edge)")
    warnings.append("PAPER MODE: backtest and holdout showed no edge for these rules")
    if not engine.notifier.channels:
        warnings.append("Telegram/Discord not configured (.env); alerts only on the dashboard")
    return {
        "mode": s.live.mode,
        "engine": {"running": engine.started_at is not None, "started_at": iso(engine.started_at),
                   "last_bar_at": iso(engine.last_bar), "feed_ok": engine.feed_ok,
                   "feed_lag_seconds": engine.feed_lag, "feed_error": engine.feed_error},
        "status": engine.status, "status_reason": engine.status_reason,
        "session": getattr(engine, "session", None),
        "price": price, "price_offset": engine.price_offset(),
        "risk": {"initial_balance": lim.initial_balance, "balance": st.balance,
                 "day_pnl": st.balance - st.day_start_balance, "daily_limit_usd": lim.daily_limit_usd,
                 "daily_buffer_usd": lim.daily_buffer_usd, "daily_loss_used_usd": st.day_loss,
                 "max_dd_usd": lim.max_dd_usd, "dd_buffer_usd": lim.dd_buffer_usd,
                 "drawdown_used_usd": st.drawdown, "signals_today": st.signals_today,
                 "max_signals_per_day": lim.max_per_day, "consecutive_losses": st.consecutive_losses,
                 "max_consecutive_losses": lim.max_consecutive_losses, "halted": st.halted,
                 "halt_reason": st.halt_reason},
        "news": {"available": news.available, "blackout": news.blackout, "reason": news.reason,
                 "next_event": {"title": news.next_event_title, "ts": iso(news.next_event_ts)}
                 if news.next_event_title else None,
                 "minutes_to_next": news.minutes_to_next, "last_refresh": iso(engine.news.last_ok)},
        "warnings": warnings,
    }


def overlays_payload(engine: LiveEngine, day: str | None) -> dict:
    det = engine.last_detections
    if det is None:
        now = pd.Timestamp.now(tz="UTC")
        fr = engine.frames(now)
        tf = engine.tf
        if fr[tf].empty:
            return {"levels": [], "fvgs": [], "order_blocks": [], "sweeps": [], "mss": [], "swings": []}
        from src.setups.detect import detect_all
        det = detect_all(fr[tf], {k: v for k, v in fr.items() if k != tf}, engine.s)
        engine.last_detections = det
    b = det.bars
    td_all = pd.Index([str(d) for d in trading_date(b.index, engine.s.data.trading_day_start).date])
    d = day or (td_all[-1] if len(td_all) else None)
    sel = td_all == d
    if not sel.any():
        return {"levels": [], "fvgs": [], "order_blocks": [], "sweeps": [], "mss": [], "swings": []}
    lo, hi = int(sel.argmax()), int(len(sel) - 1 - sel[::-1].argmax())
    t0, t1 = b.index[lo], b.index[hi] + pd.Timedelta(minutes=5)

    def t(i):
        return iso(b.index[int(i)])

    lv = det.levels
    lv = lv[(lv["known_at"] <= t1) & (lv["expires_at"] > t0)] if len(lv) else lv
    fv = det.fvgs
    fv = fv[(fv["idx"] >= lo) & (fv["idx"] <= hi)] if len(fv) else fv
    sw = det.sweeps
    sw = sw[(sw["pen_idx"] >= lo) & (sw["pen_idx"] <= hi)] if len(sw) else sw
    ms = det.mss
    ms = ms[(ms["mss_idx"] >= lo) & (ms["mss_idx"] <= hi)] if len(ms) else ms
    sg = det.swings
    sg = sg[(sg["idx"] >= lo) & (sg["idx"] <= hi)] if len(sg) and "idx" in sg else sg.iloc[0:0]
    st = det.setups
    obs = st[(st["setup_type"] == "ob_retest") & st["ob_idx"].notna()] if len(st) else st
    obs = obs[(obs["mss_idx"] >= lo) & (obs["mss_idx"] <= hi)] if len(obs) else obs

    return {
        "levels": [{"kind": r["kind"], "price": float(r["price"]), "side": r["side"], "known_at": iso(r["known_at"]),
                    "consumed_at": iso(r["consumed_at"]) if pd.notna(r.get("consumed_at")) else None}
                   for _, r in lv.iterrows()],
        "fvgs": [{"direction": r["direction"], "top": float(r["top"]), "bottom": float(r["bottom"]),
                  "start": t(r["idx"] - 2), "end": t(r["idx"]),
                  "filled_at": t(r["filled_idx"]) if pd.notna(r.get("filled_idx")) and r["filled_idx"] < len(b)
                  else None}
                 for _, r in fv.iterrows()],
        "order_blocks": [{"top": float(r["ob_top"]), "bottom": float(r["ob_bottom"]), "start": t(r["ob_idx"]),
                          "direction": "bear" if r["direction"] == "short" else "bull"} for _, r in obs.iterrows()],
        "sweeps": [{"t": t(r["pen_idx"]), "price": float(r["extreme"]), "level_kind": r["level_kind"],
                    "direction": r["direction"]} for _, r in sw.iterrows()],
        "mss": [{"t": t(r["mss_idx"]), "price": float(r["ref_price"]), "direction": r["direction"]}
                for _, r in ms.iterrows()],
        "swings": [{"t": t(r["idx"]), "price": float(r["price"]), "kind": r["kind"]} for _, r in sg.iterrows()],
    }
