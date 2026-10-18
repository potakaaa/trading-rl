"""Live rules desk (Crabel NR4 bracket, opening drive): replayed live results == backtest results."""
import numpy as np
import pandas as pd
import pytest

from src.alerts.notify import Notifier
from src.data.resample import resample
from src.data.synthetic import synthetic_m1
from src.live.engine import LiveEngine
from src.live.store import Store
from src.news.calendar import LiveNewsCalendar
from src.news.providers import NewsEvent, NewsProvider
from src.strategies.published import Cost, iter_days
from src.strategies.variants import crabel, opening_drive
from src.timeutils import NY


class Quiet(NewsProvider):
    name = "quiet"

    def fetch(self):
        return [NewsEvent(ts=pd.Timestamp("2030-01-01", tz="UTC"), title="x", currency="USD", impact="High",
                          source="t")]


class Replay:
    name = "replay"

    def __init__(self, m1):
        self.m1 = m1

    def fetch_before(self, end, limit):
        return self.m1[self.m1.index + pd.Timedelta(minutes=1) <= end].tail(limit)

    def fetch_history(self, start, end):
        d = self.m1[(self.m1.index >= start) & (self.m1.index < end)]
        return d[d.index + pd.Timedelta(minutes=1) <= end]


@pytest.fixture(scope="module")
def synth():
    return synthetic_m1("2024-01-01", "2024-04-01", seed=12)


def expected(m1, settings):
    d1 = resample(m1, "1d").sort_index()
    d1["ny_date"] = pd.DatetimeIndex(d1["end"]).tz_convert(NY).normalize().tz_localize(None)
    cost = Cost(settings.broker.typical_spread_points, settings.backtest.stop_slippage_points)
    out = {}
    for d in iter_days(m1, d1):
        for name, tr in (("crabel_nr4", crabel(d, cost, d1, "NR4", 1.0, 3.0)), ("opening_drive",
                                                                                 opening_drive(d, cost, d1))):
            if tr:
                out[(str(d.date.date()), name)] = tr[0]["r"]
    return out


def test_live_rules_desk_matches_backtest(settings, synth, tmp_path):
    settings.live.lookback_days["5m"] = 12
    settings.internal_buffers.max_signals_per_day = 9            # isolate the desk from the caps for parity
    settings.internal_buffers.max_signals_per_session = 9
    settings.internal_buffers.max_consecutive_losses = 9
    settings.internal_buffers.daily_loss_stop_fraction = 9
    exp = expected(synth, settings)
    days = sorted({d for d, _ in exp if d >= "2024-02-12"})[:3]
    assert days, "synthetic data should trigger the rule setups"
    frames = {tf: resample(synth, tf) for tf in ("1m", "5m", "15m", "1h", "4h", "1d")}

    def loader(tf, a, b):
        f = frames[tf]
        return f[(f.index >= a) & (f.index < b)]

    seen = {}
    for day in days:
        eng = LiveEngine(settings, Replay(synth), LiveNewsCalendar(Quiet(), settings), Store(tmp_path / f"{day}.db"),
                         Notifier(env={}), history_loader=loader)
        eng.on_bar_close = lambda *a, **k: None                  # only the rules desk under test
        t = pd.Timestamp(f"{day} 09:20", tz=NY).tz_convert("UTC")
        eng.bootstrap(t)
        while t < pd.Timestamp(f"{day} 16:05", tz=NY).tz_convert("UTC"):
            t += pd.Timedelta(minutes=1)
            eng.poll(t + pd.Timedelta(seconds=20))
        for sg in eng.store.signals(20):
            if sg["status"] == "closed":
                seen[(day, sg["setup_type"])] = sg["paper_r"]
    want = {k: v for k, v in exp.items() if k[0] in days}
    assert len(want) >= 2, want                                   # non-vacuous: real trades compared
    assert set(seen) == set(want), (seen, want)
    for k in want:
        assert seen[k] == pytest.approx(want[k]), k
    assert np.isfinite(list(seen.values())).all()


def test_bracket_not_issued_late_after_outage(settings, synth, tmp_path):
    frames = {tf: resample(synth, tf) for tf in ("1m", "5m", "15m", "1h", "4h", "1d")}

    def loader(tf, a, b):
        f = frames[tf]
        return f[(f.index >= a) & (f.index < b)]

    eng = LiveEngine(settings, Replay(synth), LiveNewsCalendar(Quiet(), settings), Store(tmp_path / "x.db"),
                     Notifier(env={}), history_loader=loader)
    eng.on_bar_close = lambda *a, **k: None
    t = pd.Timestamp("2024-02-12 09:20", tz=NY).tz_convert("UTC")      # an NR4 day (see the parity test)
    eng.bootstrap(t)
    eng.poll(pd.Timestamp("2024-02-12 09:45", tz=NY).tz_convert("UTC"))  # first poll only at 09:45 (outage)
    assert not [s for s in eng.store.signals(10) if s["setup_type"] == "crabel_nr4"]
    assert any(any(r.startswith("late_data") for r in b["reasons"]) for b in eng.store.blocks())
