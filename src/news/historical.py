"""Historical high-impact USD events for backtests: data/news/historical_usd_high.parquet

Sources (each event row records which one it came from):
1. FRED / ALFRED release dates (St. Louis Fed, official): CPI, NFP, PPI, GDP, Retail Sales,
   Jobless Claims, PCE, JOLTS. Actual historical publication dates, including delays such as
   government shutdowns. Times are the standard release times from settings.yaml.
2. Federal Reserve FOMC calendar pages: rate decision (14:00) + press conference (14:30).
3. Rule-based: ISM Manufacturing (1st business day) and Services (3rd business day) at 10:00.
   ISM is not on FRED; the rule matches ISM's published schedule but is marked "rule:ism".
4. Our own archive of the live ForexFactory feed (grows while the live engine runs).
5. Optional manual CSV (ts_ny,title) for anything missing, e.g. Powell testimony/speeches.

Known gap: Fed Chair speeches are not in any free official dataset. Add the ones you care about
to the manual CSV. Treating every listed release as High impact is deliberately conservative.
"""
from __future__ import annotations

import os
import re
from datetime import date

import httpx
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from src.config import PROJECT_ROOT, Settings, data_dir
from src.news.forexfactory import load_archive
from src.news.providers import NewsEvent, events_to_frame, ny_time_to_utc

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
_MONTH_IDX = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MONTH_IDX.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)})


def history_path(settings: Settings):
    return data_dir(settings) / "news" / "historical_usd_high.parquet"


# ------------------------------------------------------------------------------------- FRED

def fred_events(settings: Settings, start: date, api_key: str | None = None,
                transport: httpx.BaseTransport | None = None) -> list[NewsEvent]:
    api_key = api_key or os.getenv("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set. Get a free key at "
                           "https://fredaccount.stlouisfed.org/apikeys and put it in .env")
    h = settings.news.historical
    base = dict(api_key=api_key, file_type="json")
    out: list[NewsEvent] = []
    with httpx.Client(base_url=h.fred_api_url, timeout=30, transport=transport) as c:
        r = c.get("/releases", params={**base, "limit": 1000})
        r.raise_for_status()
        ids = {x["name"].strip().lower(): x["id"] for x in r.json()["releases"]}
        for rel in h.fred_releases:
            rid = ids.get(rel.fred_name.lower())
            if rid is None:
                raise RuntimeError(f"FRED release not found: {rel.fred_name!r}")
            r = c.get("/release/dates", params={
                **base, "release_id": rid, "realtime_start": start.isoformat(),
                "realtime_end": "9999-12-31", "include_release_dates_with_no_data": "false",
                "sort_order": "asc", "limit": 10000})
            r.raise_for_status()
            for d in r.json()["release_dates"]:
                day = pd.Timestamp(d["date"])
                if day.dayofweek >= 5:      # FRED sometimes records weekend revision dates
                    continue
                out.append(NewsEvent(ts=ny_time_to_utc(day, rel.time), title=rel.title,
                                     source=f"fred:{rid}"))
    return out


# ------------------------------------------------------------------------------------- FOMC

def parse_fomc_calendar(html: str) -> list[date]:
    """Decision days from federalreserve.gov/monetarypolicy/fomccalendars.htm."""
    days = []
    sections = re.split(r"(\d{4})\s+FOMC Meetings", html)
    # sections = [pre, year, body, year, body, ...]
    for year, body in zip(sections[1::2], sections[2::2]):
        y = int(year)
        months = re.findall(r'fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([A-Za-z/]+)', body)
        dates = re.findall(r'fomc-meeting__date[^>]*>\s*([^<]+)<', body)
        for mon, dt in zip(months, dates):
            if "notation" in dt.lower():
                continue
            nums = re.findall(r"\d{1,2}", dt)
            if not nums:
                continue
            last_month = mon.split("/")[-1].strip().lower()
            m = _MONTH_IDX.get(last_month) or _MONTH_IDX.get(last_month[:3])
            if m is None:
                continue
            days.append(date(y, m, int(nums[-1])))
    return days


_HIST_RE = re.compile(
    r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:\s*-\s*(?:(" + "|".join(MONTHS) + r")\s+)?(\d{1,2}))?"
    r",?\s+(?:\d{4}\s+)?(?:FOMC\s+)?Meeting", re.IGNORECASE)


def parse_fomc_historical(html: str, year: int) -> list[date]:
    """Decision days from fomchistorical{year}.htm ('April 30-May 1 Meeting')."""
    days = set()
    for m1, d1, m2, d2 in _HIST_RE.findall(html):
        mon = _MONTH_IDX[(m2 or m1).lower()]
        days.add(date(year, mon, int(d2 or d1)))
    return sorted(days)


def fomc_events(settings: Settings, start: date, transport: httpx.BaseTransport | None = None,
                log=print) -> list[NewsEvent]:
    f = settings.news.historical.fomc
    days: set[date] = set()
    with httpx.Client(timeout=30, transport=transport, follow_redirects=True,
                      headers={"User-Agent": "ict-signal-assistant/0.1"}) as c:
        r = c.get(f.calendar_url)
        r.raise_for_status()
        days |= set(parse_fomc_calendar(r.text))
        covered = {d.year for d in days}
        for y in range(start.year, date.today().year + 1):
            if y in covered:
                continue
            r = c.get(f.historical_url_template.format(year=y))
            if r.status_code == 200:
                days |= set(parse_fomc_historical(r.text, y))
    by_year = pd.Series(1, index=[d.year for d in days]).groupby(level=0).sum()
    for y in range(start.year, date.today().year):
        n = int(by_year.get(y, 0))
        if n < 8:
            log(f"WARNING: only {n} FOMC meetings parsed for {y} (expected 8). "
                "Check the Fed page format or add them to the manual CSV.")
    out = []
    for d in sorted(days):
        if d < start:
            continue
        out.append(NewsEvent(ts=ny_time_to_utc(d, f.statement_time), title="FOMC Statement",
                             source="fed"))
        out.append(NewsEvent(ts=ny_time_to_utc(d, f.press_conference_time),
                             title="FOMC Press Conference", source="fed"))
    return out


# ------------------------------------------------------------------------------------- rules

def nth_business_day(year: int, month: int, n: int) -> date:
    bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    first = pd.Timestamp(year, month, 1)
    d = first if bday.is_on_offset(first) else first + bday
    return (d + (n - 1) * bday).date()


def rule_events(settings: Settings, start: date, end: date) -> list[NewsEvent]:
    out = []
    for key, rule in settings.news.historical.rule_based.items():
        for p in pd.period_range(start, end, freq="M"):
            d = nth_business_day(p.year, p.month, rule.business_day)
            if start <= d <= end:
                out.append(NewsEvent(ts=ny_time_to_utc(d, rule.time), title=rule.title,
                                     source=f"rule:{key}"))
    return out


# ------------------------------------------------------------------------------------- manual

def manual_events(settings: Settings) -> list[NewsEvent]:
    p = PROJECT_ROOT / settings.news.historical.manual_csv
    if not p.exists():
        return []
    df = pd.read_csv(p)
    out = []
    for _, r in df.iterrows():
        ts = pd.Timestamp(r["ts_ny"]).tz_localize("America/New_York").tz_convert("UTC")
        out.append(NewsEvent(ts=ts, title=str(r["title"]), source="manual"))
    return out


# ------------------------------------------------------------------------------------- build

def build_history(settings: Settings, start: date | None = None, end: date | None = None,
                  log=print, fred_transport=None, fed_transport=None) -> pd.DataFrame:
    start = start or date.fromisoformat(settings.data.history_start)
    end = end or date.today()
    parts: dict[str, list[NewsEvent]] = {}
    parts["fred"] = fred_events(settings, start, transport=fred_transport)
    parts["fed"] = fomc_events(settings, start, transport=fed_transport, log=log)
    parts["rules"] = rule_events(settings, start, end)
    parts["ff_archive"] = load_archive(settings)
    parts["manual"] = manual_events(settings)
    for k, v in parts.items():
        log(f"{k:>10}: {len(v)} events")
    df = events_to_frame([e for v in parts.values() for e in v])
    df = df.drop_duplicates(subset=["ts", "title"]).reset_index(drop=True)
    p = history_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    log(f"saved {len(df)} events -> {p}")
    return df


def load_history(settings: Settings) -> pd.DataFrame:
    p = history_path(settings)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run: uv run python -m src.cli news build-history")
    return pd.read_parquet(p)
