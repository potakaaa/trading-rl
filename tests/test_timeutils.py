"""Timezone / DST handling, including the March and November switches."""
import pandas as pd

from src.timeutils import bucket_labels, in_window, is_market_open, session_of, trading_date


def utc(s):
    return pd.DatetimeIndex([pd.Timestamp(s, tz="UTC")])


def test_session_open_utc_moves_with_dst():
    # 09:30 NY = 14:30 UTC in winter (EST) and 13:30 UTC in summer (EDT)
    assert in_window(utc("2024-03-08 14:30"), "09:30", "11:30")[0]      # Fri before switch
    assert not in_window(utc("2024-03-08 13:30"), "09:30", "11:30")[0]
    assert in_window(utc("2024-03-11 13:30"), "09:30", "11:30")[0]      # Mon after March switch
    assert not in_window(utc("2024-03-11 14:30"), "09:30", "09:31")[0]
    assert in_window(utc("2024-11-01 13:30"), "09:30", "11:30")[0]      # Fri before Nov switch
    assert in_window(utc("2024-11-04 14:30"), "09:30", "11:30")[0]      # Mon after Nov switch
    assert not in_window(utc("2024-11-04 13:30"), "09:30", "11:30")[0]


def test_window_crossing_midnight():
    assert in_window(utc("2024-05-07 01:00"), "20:00", "00:00")[0]      # 21:00 NY
    assert not in_window(utc("2024-05-07 05:00"), "20:00", "00:00")[0]  # 01:00 NY


def test_trading_date_rolls_at_18_ny():
    # Sunday 18:00 NY (22:00 UTC summer) belongs to Monday
    assert str(trading_date(utc("2024-06-09 22:00"))[0].date()) == "2024-06-10"
    assert str(trading_date(utc("2024-06-09 21:59"))[0].date()) == "2024-06-09"
    # winter: 18:00 NY = 23:00 UTC
    assert str(trading_date(utc("2024-01-08 23:00"))[0].date()) == "2024-01-09"
    assert str(trading_date(utc("2024-01-08 22:59"))[0].date()) == "2024-01-08"


def test_market_hours():
    assert not is_market_open(utc("2024-06-08 15:00"))[0]   # Saturday
    assert not is_market_open(utc("2024-06-09 21:00"))[0]   # Sunday 17:00 NY
    assert is_market_open(utc("2024-06-09 22:00"))[0]       # Sunday 18:00 NY
    assert not is_market_open(utc("2024-06-11 21:30"))[0]   # Tue 17:30 NY break
    assert not is_market_open(utc("2024-06-14 21:00"))[0]   # Friday 17:00 NY
    assert is_market_open(utc("2024-06-14 20:59"))[0]


def test_bucket_labels_4h_and_daily_across_dst():
    # 4h buckets start at 18:00 NY -> 10:00 NY bucket covers 10:00-14:00 NY
    t = utc("2024-03-08 16:15")  # 11:15 EST
    assert bucket_labels(t, "4h")[0] == pd.Timestamp("2024-03-08 15:00", tz="UTC")  # 10:00 EST
    t = utc("2024-03-11 15:15")  # 11:15 EDT
    assert bucket_labels(t, "4h")[0] == pd.Timestamp("2024-03-11 14:00", tz="UTC")  # 10:00 EDT
    t = utc("2024-11-04 16:15")  # 11:15 EST after November switch
    assert bucket_labels(t, "4h")[0] == pd.Timestamp("2024-11-04 15:00", tz="UTC")
    # daily bar opens 18:00 NY the previous evening
    assert bucket_labels(utc("2024-03-11 15:15"), "1d")[0] == pd.Timestamp("2024-03-10 22:00", tz="UTC")
    assert bucket_labels(utc("2024-03-08 16:15"), "1d")[0] == pd.Timestamp("2024-03-07 23:00", tz="UTC")
    assert bucket_labels(utc("2024-11-04 16:15"), "1d")[0] == pd.Timestamp("2024-11-03 23:00", tz="UTC")


def test_session_of():
    from src.config import load_settings
    s = load_settings()
    idx = pd.DatetimeIndex(pd.to_datetime(["2024-06-10 13:45", "2024-06-10 16:00", "2024-06-10 18:00"], utc=True))
    assert list(session_of(idx, s.sessions)) == ["ny_am", None, "ny_pm"]
