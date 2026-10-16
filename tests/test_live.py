"""Phase 5 tests: live feed parsing, store, alerts, and live-vs-backtest parity (same code, same answers)."""
import httpx
import numpy as np
import pandas as pd
import pytest

from src.alerts.notify import Notifier, format_signal
from src.data.dukascopy_live import DukascopyLiveProvider, FeedError, parse_jsonp
from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.live.engine import LiveEngine, trading_day_start_utc
from src.live.store import Store
from src.news.calendar import LiveNewsCalendar
from src.news.providers import NewsEvent, NewsProvider
from src.setups.detect import detect_all

NY = "America/New_York"


def ny(s):
    return pd.Timestamp(s, tz=NY).tz_convert("UTC")


# ------------------------------------------------------------------------------- live feed

def test_parse_jsonp_newest_first_to_sorted_frame():
    df = parse_jsonp("_cb([[1792154340000,2,3,1,2.5,0.07],[1792154280000,1,2,0.5,1.5,0.08]]);")
    assert list(df.index) == [pd.Timestamp(1792154280000, unit="ms", tz="UTC"),
                              pd.Timestamp(1792154340000, unit="ms", tz="UTC")]
    assert df.iloc[1].to_dict() == {"open": 2, "high": 3, "low": 1, "close": 2.5, "volume": 0.07}


def test_parse_jsonp_empty_and_garbage():
    assert parse_jsonp("_cb([null]);").empty
    with pytest.raises(FeedError):
        parse_jsonp("<html>rate limited</html>")


def test_provider_errors_are_feed_errors(settings):
    down = DukascopyLiveProvider(settings, transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(FeedError):
        down.fetch_before(pd.Timestamp("2026-10-16 13:00", tz="UTC"), 10)


def test_provider_pages_backwards(settings):
    settings.live.dukascopy_live.max_bars_per_request = 3
    base = pd.Timestamp("2026-10-16 13:00", tz="UTC")

    def handler(req: httpx.Request):
        end = pd.Timestamp(int(req.url.params["timestamp"]), unit="ms", tz="UTC")
        lim = int(req.url.params["limit"])
        rows = [[int((end - pd.Timedelta(minutes=k)).timestamp() * 1000), 1, 2, 0.5, 1.5, 0.1]
                for k in range(1, lim + 1)]
        return httpx.Response(200, text=f"_cb({rows});".replace("'", ""))

    p = DukascopyLiveProvider(settings, transport=httpx.MockTransport(handler))
    df = p.fetch_history(base - pd.Timedelta(minutes=7), base)
    assert len(df) == 7
    assert df.index[0] == base - pd.Timedelta(minutes=7) and df.index[-1] == base - pd.Timedelta(minutes=1)


# ---------------------------------------------------------------------------- store/alerts

def test_store_roundtrip(tmp_path):
    st = Store(tmp_path / "x.db")
    sid = st.add_signal({"setup_key": "k", "created_at": "2026-10-16T13:45:00Z", "trading_day": "2026-10-16",
                         "status": "active"})
    assert st.has_setup("k") and st.signal(sid)["status"] == "active"
    st.add_block("b", pd.Timestamp("2026-10-16 13:50", tz="UTC"), "2026-10-16", "ob_retest", "long", ["news_blackout"])
    assert st.blocks("2026-10-16")[0]["reasons"] == ["news_blackout"] and st.has_setup("b")
    st.set("price_offset", -1.5)
    assert Store(tmp_path / "x.db").get("price_offset") == -1.5          # persisted


def test_notifier_skips_unconfigured_and_never_raises():
    n = Notifier(env={})
    n.send("hello")
    assert n.channels == [] and n.sent == []
    calls = []

    def handler(req):
        calls.append(str(req.url))
        return httpx.Response(500 if "discord" in str(req.url) else 200, json={})

    n = Notifier(env={"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "1",
                      "DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/x"},
                 transport=httpx.MockTransport(handler))
    n.send("signal!", "signal")
    assert len(calls) == 2 and [c for c, _ in n.sent] == ["telegram"] and "discord" in n.errors[0]


def test_format_signal_applies_offset():
    s = {"direction": "short", "setup_type": "sweep_mss_fvg", "session": "ny_am", "entry": 100.0, "stop": 110.0,
         "target": 80.0, "rr": 2.0, "risk_pts": 10.0, "lots": 1.25, "risk_usd": 12.5,
         "expires_at": "2026-10-16T14:15:00Z", "confidence_note": "no model", "reason": "test"}
    txt = format_signal(s, offset=-2.0)
    assert "PAPER" in txt and "Entry 98.0" in txt and "SL 108.0" in txt and "TP 78.0" in txt


def test_trading_day_start_dst():
    assert trading_day_start_utc(ny("2024-07-10 10:00"), "18:00") == pd.Timestamp("2024-07-09 22:00", tz="UTC")
    assert trading_day_start_utc(ny("2024-01-10 10:00"), "18:00") == pd.Timestamp("2024-01-09 23:00", tz="UTC")
    assert trading_day_start_utc(ny("2024-01-10 19:00"), "18:00") == pd.Timestamp("2024-01-10 23:00", tz="UTC")


# ------------------------------------------------------------------------------ parity

class ReplayProvider:
    """Serves stored 1m bars as if live: only bars complete at the requested time."""
    name = "replay"

    def __init__(self, m1: pd.DataFrame):
        self.m1 = m1

    def fetch_before(self, end, limit):
        d = self.m1[self.m1.index + pd.Timedelta(minutes=1) <= end]
        return d.tail(limit)

    def fetch_history(self, start, end):
        d = self.m1[(self.m1.index >= start) & (self.m1.index < end)]
        return d[d.index + pd.Timedelta(minutes=1) <= end]


class QuietNews(NewsProvider):
    name = "quiet"

    def fetch(self):
        return [NewsEvent(ts=pd.Timestamp("2030-01-01", tz="UTC"), title="none", currency="USD", impact="High",
                          source="test")]


def _engine_for(settings, m1, tmp_path):
    frames = {tf: resample(m1, tf, settings.data.trading_day_start, settings.data.market_close)
              for tf in ("1m", "5m", "15m", "1h", "4h", "1d")}

    def loader(tf, start, end):
        f = frames[tf]
        return f[(f.index >= start) & (f.index < end)]

    news = LiveNewsCalendar(QuietNews(), settings)
    return LiveEngine(settings, ReplayProvider(m1), news, Store(tmp_path / "live.db"), Notifier(env={}),
                      history_loader=loader)


@pytest.fixture(scope="module")
def synthetic():
    return synthetic_m1("2024-01-01", "2024-04-01", seed=3)


def test_live_replay_matches_backtest_detection(settings, synthetic, tmp_path):
    """Replay day by day through the live engine; every valid setup the full-history backtest finds
    for those days must reach the guard at the same time with the same prices, and nothing else."""
    settings.live.lookback_days["5m"] = 12
    s = settings
    tf_frames = {tf: resample(synthetic, tf, s.data.trading_day_start, s.data.market_close)
                 for tf in ("5m", "15m", "4h", "1d")}
    full = detect_all(tf_frames["5m"], {k: v for k, v in tf_frames.items() if k != "5m"}, s).setups
    valid = full[full["status"] == "valid"]
    assert len(valid) > 0, "synthetic series should produce valid setups"
    # pick up to 2 NY dates (after enough warm-up history) that have valid setups
    days = sorted({t.tz_convert(NY).date() for t in valid["armed_at"] if t >= pd.Timestamp("2024-02-01", tz="UTC")})[:2]
    assert days
    for day in days:
        eng = _engine_for(s, synthetic, tmp_path / str(day))
        seen = []
        eng.consider = lambda row, now, _seen=seen: _seen.append((row["armed_at"], row["setup_type"],
                                                                   row["direction"], round(row["entry"], 6),
                                                                   round(row["stop"], 6), round(row["target"], 6)))
        t = ny(f"{day} 09:00")
        eng.bootstrap(t)
        end = ny(f"{day} 16:10")
        while t < end:
            t += pd.Timedelta(minutes=1)
            eng.poll(t + pd.Timedelta(seconds=20))
        exp = valid[[a.tz_convert(NY).date() == day for a in valid["armed_at"]]]
        expected = sorted((r.armed_at, r.setup_type, r.direction, round(r.entry, 6), round(r.stop, 6),
                           round(r.target, 6)) for r in exp.itertuples())
        assert expected, f"{day}: no expected setups (vacuous comparison)"
        assert sorted(seen) == expected, f"{day}: live {sorted(seen)} != backtest {expected}"


def test_live_engine_signals_tracks_paper_and_blocks(settings, synthetic, tmp_path):
    """End to end on one replayed day: signals stored, paper outcome resolved, caps enforced."""
    settings.live.lookback_days["5m"] = 12
    s = settings
    tf5 = resample(synthetic, "5m")
    full = detect_all(tf5, {tf: resample(synthetic, tf) for tf in ("15m", "4h", "1d")}, s).setups
    v = full[(full["status"] == "valid") & (full["armed_at"] >= pd.Timestamp("2024-02-01", tz="UTC"))]
    day = v["armed_at"].iloc[0].tz_convert(NY).date()
    eng = _engine_for(s, synthetic, tmp_path)
    t = ny(f"{day} 09:00")
    eng.bootstrap(t)
    while t < ny(f"{day} 16:10"):
        t += pd.Timedelta(minutes=1)
        eng.poll(t + pd.Timedelta(seconds=20))
    sigs = eng.store.signals(50)
    blocks = eng.store.blocks()
    assert len(sigs) + len(blocks) >= 1
    assert all(sg["status"] in ("closed", "expired", "cancelled") for sg in sigs)   # all resolved by 16:10
    assert len(sigs) <= s.internal_buffers.max_signals_per_day
    for sg in sigs:
        assert sg["lots"] > 0 and sg["risk_usd"] <= 12.5 + 1e-9
        if sg["status"] == "closed":
            assert sg["paper_outcome"] in ("tp", "sl", "session_end", "news_flat") and np.isfinite(sg["paper_r"])
    assert eng.status in ("OUTSIDE-SESSION", "STOPPED-FOR-DAY", "ACTIVE")


def test_feed_failure_is_fail_safe(settings, synthetic, tmp_path):
    eng = _engine_for(settings, synthetic, tmp_path)
    eng.bootstrap(ny("2024-02-14 10:00"))
    assert eng.status in ("ACTIVE", "PAUSED-NEWS")

    def boom(*a, **k):
        raise FeedError("HTTP 429")

    eng.provider.fetch_before = boom
    eng.poll(ny("2024-02-14 10:01:20"))
    assert eng.status == "NO-DATA" and not eng.feed_ok             # no signals immediately
    assert not any("disconnected" in e["text"] for e in eng.store.events())   # one blip: no alert yet
    eng.poll(ny("2024-02-14 10:01:35"))
    assert any("disconnected" in e["text"] for e in eng.store.events())


def test_news_failure_is_fail_safe(settings, synthetic, tmp_path):
    class Broken(NewsProvider):
        name = "broken"

        def fetch(self):
            raise RuntimeError("rate limited")

    eng = _engine_for(settings, synthetic, tmp_path)
    eng.news = LiveNewsCalendar(Broken(), settings)
    eng.bootstrap(ny("2024-02-14 10:00"))
    assert eng.status == "NO-NEWS"


# ------------------------------------------------------------------------------------ API

@pytest.fixture
def api(settings, synthetic, tmp_path):
    from fastapi.testclient import TestClient

    from src.api.app import create_app
    settings.live.lookback_days["5m"] = 12
    eng = _engine_for(settings, synthetic, tmp_path)
    t = ny("2024-02-12 09:00")
    eng.bootstrap(t)
    while t < ny("2024-02-12 11:00"):
        t += pd.Timedelta(minutes=1)
        eng.poll(t + pd.Timedelta(seconds=20))
    with TestClient(create_app(settings, eng, run_engine=False)) as c:
        yield c, eng


def test_api_status_and_bars(api):
    c, eng = api
    s = c.get("/api/status").json()
    assert s["mode"] == "paper" and s["status"] in ("ACTIVE", "PAUSED-NEWS", "STOPPED-FOR-DAY", "OUTSIDE-SESSION")
    assert s["risk"]["daily_limit_usd"] == 100 and any("PAPER" in w for w in s["warnings"])
    for tf in ("1m", "5m", "15m"):
        b = c.get(f"/api/bars?tf={tf}&limit=50").json()
        assert len(b["bars"]) == 50 and set(b["bars"][0]) == {"t", "o", "h", "l", "c"}
        assert b["bars"][0]["t"] < b["bars"][-1]["t"]
    assert c.get("/api/bars?tf=4h").status_code == 400


def test_api_overlays_signals_and_journal_action(api):
    c, eng = api
    ov = c.get("/api/overlays").json()
    assert set(ov) == {"levels", "fvgs", "order_blocks", "sweeps", "mss", "swings"} and ov["levels"]
    sigs = c.get("/api/signals").json()["signals"]
    assert sigs, "the replayed morning should contain the 2024-02-12 setup"
    sg = sigs[0]
    assert sg["entry_broker"] == sg["entry"] and "paper" in sg
    c.post("/api/price-offset", json={"offset": -2.0})
    assert c.get("/api/signals").json()["signals"][0]["entry_broker"] == pytest.approx(sg["entry"] - 2.0)
    fill = sg["entry"]
    exit_ = fill - 10 if sg["direction"] == "short" else fill + 10
    r = c.post(f"/api/signals/{sg['id']}/action", json={"action": "taken", "fill_price": fill,
                                                         "exit_price": exit_, "lots": 1.0}).json()
    assert r["action"] == "taken" and r["pnl_usd"] == pytest.approx(10.0)
    j = c.get("/api/journal?action=taken").json()
    assert j["stats"]["taken"] == 1 and j["stats"]["total_pnl_usd"] == pytest.approx(10.0)
    assert c.post("/api/signals/999/action", json={"action": "taken"}).status_code == 404
    assert c.post(f"/api/signals/{sg['id']}/action", json={"action": "maybe"}).status_code == 400


def test_api_account_settings_news_ws(api):
    c, eng = api
    assert c.post("/api/account/balance", json={"balance": 2400}).json()["risk"]["balance"] == pytest.approx(
        2400 + sum((x["paper_pnl_usd"] or 0) for x in eng.store.signals(100) if x["paper_exit_time"]
                   and pd.Timestamp(x["paper_exit_time"]) > pd.Timestamp.now(tz="UTC")))
    eng.store.set("halted", True)
    assert c.get("/api/status").json()["risk"]["halted"] is True
    assert c.post("/api/account/reset").json()["risk"]["halted"] is False
    st = c.get("/api/settings").json()
    assert st["account"]["initial_balance"] == 2500 and "FRED" not in str(st)
    n = c.get("/api/news").json()
    # the replayed calendar was loaded at a 2024 clock, so against today's clock it is stale -> fail-safe
    assert n["available"] is False and n["events"][0]["blackout_start"] < n["events"][0]["ts"]
    assert "gate" in c.get("/api/backtest").json()
    with c.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status" and msg["data"]["mode"] == "paper"


def test_news_falls_back_to_recent_archive_but_not_old_one(settings, synthetic, tmp_path):
    import json as _json

    from src.news.forexfactory import ForexFactoryProvider, archive_dir
    d = archive_dir(settings)
    d.mkdir(parents=True)
    feed = [{"title": "CPI m/m", "country": "USD", "impact": "High", "date": "2024-02-14T08:30:00-05:00"}]
    (d / "ff_2024-02-14T1450.json").write_text(_json.dumps(feed))
    down = ForexFactoryProvider(settings, transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    eng = _engine_for(settings, synthetic, tmp_path)
    eng.news = LiveNewsCalendar(down, settings)
    eng.bootstrap(ny("2024-02-14 10:00"))                       # snapshot fetched 09:50 NY -> fresh
    assert eng.status != "NO-NEWS" and eng.news.calendar is not None
    eng2 = _engine_for(settings, synthetic, tmp_path / "b")
    eng2.news = LiveNewsCalendar(down, settings)
    eng2.bootstrap(ny("2024-02-14 15:00"))                      # 5 h later -> too old -> fail-safe
    assert eng2.status == "NO-NEWS"
    assert any("unavailable at startup" in e["text"] for e in eng2.store.events())


def test_provider_retries_once_on_connection_error(settings):
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("handshake timed out")
        return httpx.Response(200, text="_cb([[1792154340000,2,3,1,2.5,0.07]]);")

    p = DukascopyLiveProvider(settings, transport=httpx.MockTransport(handler))
    df = p.fetch_before(pd.Timestamp("2026-10-16 13:00", tz="UTC"), 5)
    assert len(calls) == 2 and len(df) == 1


def test_setup_seen_late_after_outage_is_blocked(settings, synthetic, tmp_path):
    """Replay 2024-02-12 but deliver the 10:05 NY close only 5 minutes late (feed outage)."""
    settings.live.lookback_days["5m"] = 12
    eng = _engine_for(settings, synthetic, tmp_path)
    t = ny("2024-02-12 09:00")
    eng.bootstrap(t)
    outage = (ny("2024-02-12 10:03"), ny("2024-02-12 10:12"))
    real_fetch = eng.provider.fetch_before
    while t < ny("2024-02-12 11:00"):
        t += pd.Timedelta(minutes=1)
        if outage[0] <= t < outage[1]:
            eng.provider.fetch_before = lambda *a, **k: (_ for _ in ()).throw(FeedError("down"))
        else:
            eng.provider.fetch_before = real_fetch
        eng.poll(t + pd.Timedelta(seconds=20))
    blocks = eng.store.blocks()
    assert any(any(r.startswith("late_data") for r in b["reasons"]) for b in blocks), blocks
    assert not eng.store.signals(10)        # the 10:05 setup must not become a signal
