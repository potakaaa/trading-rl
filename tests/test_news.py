from datetime import date

import httpx
import pandas as pd
import pytest

from src.news.calendar import LiveNewsCalendar, NewsCalendar
from src.news.forexfactory import ForexFactoryProvider, NewsFeedError, load_archive, parse_ff
from src.news.historical import (
    build_history,
    nth_business_day,
    parse_fomc_calendar,
    parse_fomc_historical,
    rule_events,
)
from src.news.providers import NewsEvent, events_to_frame, ny_time_to_utc

FF = [
    {"title": "Non-Farm Employment Change", "country": "USD", "date": "2024-03-08T08:30:00-05:00",
     "impact": "High", "forecast": "200K", "previous": "353K"},
    {"title": "CPI m/m", "country": "USD", "date": "2024-03-12T08:30:00-04:00", "impact": "High"},
    {"title": "Final Manufacturing PMI", "country": "USD", "date": "2024-03-01T09:45:00-05:00", "impact": "Low"},
    {"title": "ECB Press Conference", "country": "EUR", "date": "2024-03-07T08:45:00-05:00", "impact": "High"},
]


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def test_parse_ff_filters_usd_high_and_converts_to_utc():
    ev = parse_ff(FF)
    assert [e.title for e in ev] == ["Non-Farm Employment Change", "CPI m/m"]
    assert ev[0].ts == utc("2024-03-08 13:30")   # EST
    assert ev[1].ts == utc("2024-03-12 12:30")   # EDT, after the March switch


def test_parse_ff_rejects_non_list():
    with pytest.raises(NewsFeedError):
        parse_ff({"error": "rate limited"})


def test_parse_ff_rejects_empty_list():
    with pytest.raises(NewsFeedError):
        parse_ff([])


def test_ff_provider_archives_and_fails_on_bad_response(settings):
    ok = httpx.MockTransport(lambda r: httpx.Response(200, json=FF))
    assert len(ForexFactoryProvider(settings, transport=ok).fetch()) == 2
    assert len(load_archive(settings)) == 2
    html = httpx.MockTransport(lambda r: httpx.Response(200, text="<html>Request denied</html>"))
    with pytest.raises(NewsFeedError):
        ForexFactoryProvider(settings, transport=html).fetch()
    down = httpx.MockTransport(lambda r: httpx.Response(503))
    with pytest.raises(NewsFeedError):
        ForexFactoryProvider(settings, transport=down).fetch()


def cal(before=5, after=5):
    ev = events_to_frame(parse_ff(FF))
    return NewsCalendar(ev, before, after)


def test_blackout_window_edges():
    c = cal(before=5, after=10)
    nfp = utc("2024-03-08 13:30")
    assert not c.check(nfp - pd.Timedelta(minutes=6)).blackout
    assert c.check(nfp - pd.Timedelta(minutes=5)).blackout
    assert c.check(nfp).blackout
    assert c.check(nfp + pd.Timedelta(minutes=10)).blackout
    assert not c.check(nfp + pd.Timedelta(minutes=11)).blackout
    st = c.check(nfp - pd.Timedelta(minutes=30))
    assert st.next_event_title == "Non-Farm Employment Change" and st.minutes_to_next == 30


def test_blackout_dst_both_sides():
    c = cal(before=2, after=2)
    # CPI 08:30 NY on 2024-03-12 (EDT) = 12:30 UTC, not 13:30
    assert c.check(utc("2024-03-12 12:31")).blackout
    assert not c.check(utc("2024-03-12 13:31")).blackout


def test_blackout_mask_matches_check():
    c = cal(before=5, after=5)
    idx = pd.date_range("2024-03-08 13:00", "2024-03-08 14:00", freq="1min", tz="UTC")
    mask = c.blackout_mask(idx)
    assert list(mask) == [c.check(t).blackout for t in idx]
    assert mask.sum() == 11


def test_unconfigured_window_refuses():
    with pytest.raises(ValueError, match="TODO"):
        NewsCalendar(events_to_frame([]), None, 5)


class FlakyProvider:
    def __init__(self):
        self.ok = True

    def fetch(self):
        if not self.ok:
            raise NewsFeedError("down")
        return parse_ff(FF)


def test_live_calendar_fail_safe(settings):
    settings.news.blackout_minutes_before = 5
    settings.news.blackout_minutes_after = 5
    p = FlakyProvider()
    live = LiveNewsCalendar(p, settings)
    t0 = utc("2024-03-08 12:00")
    assert live.status(t0).blocks_signals            # never loaded -> blocked
    assert live.refresh(t0)
    assert not live.status(t0).blocks_signals
    p.ok = False
    assert not live.refresh(t0 + pd.Timedelta(minutes=30))
    assert not live.status(t0 + pd.Timedelta(minutes=60)).blocks_signals   # still fresh enough
    st = live.status(t0 + pd.Timedelta(minutes=settings.news.stale_after_minutes + 1))
    assert st.blocks_signals and not st.available and "down" in st.reason


def test_nth_business_day_skips_holidays_and_weekends():
    assert nth_business_day(2024, 1, 1) == date(2024, 1, 2)    # Jan 1 holiday
    assert nth_business_day(2024, 9, 1) == date(2024, 9, 3)    # Sun 1st, Labor Day Mon 2nd
    assert nth_business_day(2024, 6, 3) == date(2024, 6, 5)


def test_rule_events_ism_times(settings):
    ev = rule_events(settings, date(2024, 1, 1), date(2024, 1, 31))
    titles = {e.title: e.ts for e in ev}
    assert titles["ISM Manufacturing PMI"] == utc("2024-01-02 15:00")   # 10:00 EST
    assert titles["ISM Services PMI"] == utc("2024-01-04 15:00")


FED_HTML = """
<h4><a id="1">2024 FOMC Meetings</a></h4>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
<div class="fomc-meeting__date col-xs-4">30-31</div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>March</strong></div>
<div class="fomc-meeting__date col-xs-4">19-20*</div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>Apr/May</strong></div>
<div class="fomc-meeting__date col-xs-4">30-1</div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>August</strong></div>
<div class="fomc-meeting__date col-xs-4">16 (notation vote)</div></div>
<h4><a id="2">2023 FOMC Meetings</a></h4>
<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>December</strong></div>
<div class="fomc-meeting__date col-xs-4">12-13*</div></div>
"""


def test_parse_fomc_calendar():
    assert parse_fomc_calendar(FED_HTML) == [date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
                                             date(2023, 12, 13)]


def test_parse_fomc_historical():
    html = "<h5>January 29-30 Meeting</h5> ... <h5>April 30-May 1 Meeting</h5> <p>March 15, 2020 Meeting</p>"
    assert parse_fomc_historical(html, 2019) == [date(2019, 1, 30), date(2019, 3, 15), date(2019, 5, 1)]


def test_build_history_with_mocks(settings, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")

    def fred(req: httpx.Request):
        if req.url.path.endswith("/releases"):
            return httpx.Response(200, json={"releases": [
                {"id": i, "name": r.fred_name} for i, r in enumerate(settings.news.historical.fred_releases)]})
        rid = int(req.url.params["release_id"])
        dates = [{"date": "2024-03-12"}, {"date": "2024-03-16"}] if rid == 0 else []
        return httpx.Response(200, json={"release_dates": dates})

    fed = httpx.MockTransport(lambda r: httpx.Response(200, text=FED_HTML) if "fomccalendars" in str(r.url)
                              else httpx.Response(404))
    df = build_history(settings, date(2024, 1, 1), date(2024, 3, 31), log=lambda *_: None,
                       fred_transport=httpx.MockTransport(fred), fed_transport=fed)
    cpi = df[df.title == "CPI"]
    assert list(cpi.ts) == [utc("2024-03-12 12:30")]           # weekend date dropped, EDT time
    fomc = df[df.title == "FOMC Statement"]
    assert utc("2024-03-20 18:00") in list(fomc.ts)              # 14:00 EDT
    assert (df.source == "rule:ism_manufacturing").sum() == 3


def test_ny_time_to_utc_dst():
    assert ny_time_to_utc("2024-11-01", "08:30") == utc("2024-11-01 12:30")
    assert ny_time_to_utc("2024-11-04", "08:30") == utc("2024-11-04 13:30")


def test_events_to_frame_empty():
    assert events_to_frame([]).empty
    assert len(events_to_frame([NewsEvent(ts=utc("2024-01-01"), title="x")])) == 1


def test_parse_fomc_calendar_real_2024_page_snippet():
    """Real federalreserve.gov markup (saved 2026-10-16). 2024 decision days are public record."""
    from pathlib import Path
    html = (Path(__file__).parent / "fixtures" / "fomc_calendar_2024_snippet.html").read_text()
    assert [d for d in parse_fomc_calendar(html) if d.year == 2024] == [
        date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
        date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18)]
