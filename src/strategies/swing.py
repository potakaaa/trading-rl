"""Round-3 swing strategies, exactly as pre-registered in docs/STRATEGIES_R3.md (frozen 2026-10-17).

Signals are computed at the DAILY close (bar complete 17:00 NY) from complete daily bars only.
Entries and rule exits are market orders at 09:30 NY on the next trading day; stops (and the 1R
target of rsi2_revert) are live orders simulated on 1m bars with backtest.fills.simulate. Positions
can be held overnight and over weekends; swap is charged per calendar night held.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.fills import M1, Costs, simulate
from src.config import Settings
from src.features.basics import atr
from src.timeutils import NY

SWING_STRATEGIES = ["donchian", "ma_trend", "rsi2_revert", "tsmom", "turn_of_month", "overnight"]
INDEX_ONLY = {"turn_of_month", "overnight"}
INDICES = {"US100", "US500"}


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out[(dn == 0) & (up > 0)] = 100.0          # only up moves: RSI is 100, not undefined
    out[(dn == 0) & (up == 0)] = 50.0
    return out


def daily_indicators(d1: pd.DataFrame) -> pd.DataFrame:
    """Indicators on complete daily bars. Row k uses bars 0..k only."""
    d = d1.sort_index().copy()
    d["atr20"] = atr(d, 20)
    d["hh20"] = d["high"].rolling(20).max().shift(1)         # previous 20 days, excluding today
    d["ll20"] = d["low"].rolling(20).min().shift(1)
    d["hh10"] = d["high"].rolling(10).max().shift(1)
    d["ll10"] = d["low"].rolling(10).min().shift(1)
    d["sma50"] = d["close"].rolling(50).mean()
    d["sma200"] = d["close"].rolling(200).mean()
    d["sma5"] = d["close"].rolling(5).mean()
    d["rsi2"] = rsi(d["close"], 2)
    d["ret63"] = d["close"] / d["close"].shift(63) - 1
    d["ret20"] = d["close"] / d["close"].shift(20) - 1
    d["ny_date"] = pd.DatetimeIndex(d["end"]).tz_convert(NY).normalize().tz_localize(None)
    d["weekday"] = d["ny_date"].dt.weekday
    return d


@dataclass
class Clock:
    """Timestamps (ns) of the 09:30 / 09:35 / 15:55 NY 1m bars: execution times for the rules."""
    t0930: np.ndarray
    t0935: np.ndarray
    t1555: np.ndarray

    @classmethod
    def from_m1(cls, m1: M1) -> Clock:
        idx = pd.DatetimeIndex(m1.ts, tz="UTC").tz_convert(NY)
        mins = idx.hour * 60 + idx.minute
        return cls(m1.ts[mins == 570], m1.ts[mins == 575], m1.ts[mins == 955])

    @staticmethod
    def next_after(arr: np.ndarray, t_ns: int) -> int | None:
        k = int(np.searchsorted(arr, t_ns, side="right"))
        return int(arr[k]) if k < len(arr) else None


def swap_pts(spec, price: float, nights: int, mult: float = 1.0) -> float:
    if spec.swap.type == "pct":
        return mult * nights * price * spec.swap.annual_pct / 100 / 360
    return mult * nights * spec.swap.per_night


def _nights(t_in: pd.Timestamp, t_out: pd.Timestamp) -> int:
    a = t_in.tz_convert(NY).normalize()
    b = t_out.tz_convert(NY).normalize()
    return max(0, (b - a).days)


# ------------------------------------------------------------------------ signal rules

def _entry_signal(name: str, d: pd.DataFrame, k: int, inst: str):
    """Direction to open at the close of daily bar k (None = no signal)."""
    r = d.iloc[k]
    if name == "donchian":
        if pd.notna(r["hh20"]) and r["close"] > r["hh20"]:
            return "long"
        if pd.notna(r["ll20"]) and r["close"] < r["ll20"]:
            return "short"
    elif name == "ma_trend":
        if k == 0 or pd.isna(r["sma200"]) or pd.isna(d["sma200"].iloc[k - 1]):
            return None
        p = d.iloc[k - 1]
        if p["sma50"] <= p["sma200"] and r["sma50"] > r["sma200"]:
            return "long"
        if p["sma50"] >= p["sma200"] and r["sma50"] < r["sma200"]:
            return "short"
    elif name == "rsi2_revert":
        if pd.isna(r["sma200"]) or pd.isna(r["rsi2"]):
            return None
        if r["close"] > r["sma200"] and r["rsi2"] < 10:
            return "long"
        if r["close"] < r["sma200"] and r["rsi2"] > 90:
            return "short"
    elif name == "tsmom":
        if r["weekday"] != 4 or pd.isna(r["ret63"]) or r["ret63"] == 0:
            return None
        return "long" if r["ret63"] > 0 else "short"
    elif name == "turn_of_month":
        if inst not in INDICES or k + 2 >= len(d):
            return None
        # second-to-last trading day of the month (the exchange calendar is known in advance)
        m = d["ny_date"].iloc[k].month
        if d["ny_date"].iloc[k + 1].month == m and d["ny_date"].iloc[k + 2].month != m:
            return "long"
    return None


def _rule_exit_k(name: str, d: pd.DataFrame, k_entry: int, direction: str) -> int | None:
    """Index of the daily close at which the rule says exit (exit executes next 09:30)."""
    n = len(d)
    if name == "donchian":
        for j in range(k_entry + 1, n):
            r = d.iloc[j]
            if direction == "long" and pd.notna(r["ll10"]) and r["close"] < r["ll10"]:
                return j
            if direction == "short" and pd.notna(r["hh10"]) and r["close"] > r["hh10"]:
                return j
    elif name == "ma_trend":
        for j in range(k_entry + 1, n):
            r = d.iloc[j]
            below = r["sma50"] < r["sma200"]
            if (direction == "long" and below) or (direction == "short" and r["sma50"] > r["sma200"]):
                return j
    elif name == "rsi2_revert":
        return min(k_entry + 10, n - 1)
    elif name == "tsmom":
        for j in range(k_entry + 1, n):
            r = d.iloc[j]
            if r["weekday"] == 4 and pd.notna(r["ret63"]) and ((r["ret63"] > 0) != (direction == "long")):
                return j
    elif name == "turn_of_month":
        m_new = None
        count = 0
        for j in range(k_entry + 1, n):
            mo = d["ny_date"].iloc[j].month
            if m_new is None and mo != d["ny_date"].iloc[k_entry].month:
                m_new = mo
            if m_new is not None and mo == m_new:
                count += 1
                if count == 3:
                    return j
    return None


STOP_ATR = {"donchian": 2.0, "ma_trend": 3.0, "rsi2_revert": 2.0, "tsmom": 3.0, "turn_of_month": 2.0,
            "overnight": 1.0}


# ----------------------------------------------------------------------------- trades

def run_strategy(name: str, inst: str, spec, d1: pd.DataFrame, m1: M1, clock: Clock,
                 start: pd.Timestamp, end: pd.Timestamp | None, cost_mult: float = 1.0) -> list[dict]:
    if name in INDEX_ONLY and inst not in INDICES:
        return []
    d = daily_indicators(d1)
    costs = Costs(spec.spread * cost_mult, spec.stop_slippage * cost_mult)
    ends = pd.DatetimeIndex(d["end"]).as_unit("ns").asi8
    trades: list[dict] = []
    flat_from = 0                                              # ns time from which a new entry is allowed
    if name == "overnight":
        return _run_overnight(inst, spec, d, m1, clock, start, end, costs, cost_mult)
    for k in range(len(d)):
        sig_t = ends[k]
        if sig_t < start.value or (end is not None and sig_t >= end.value):
            continue
        direction = _entry_signal(name, d, k, inst)
        if direction is None:
            continue
        t_in = Clock.next_after(clock.t0930, sig_t)
        if t_in is None or t_in < flat_from or not np.isfinite(d["atr20"].iloc[k]):
            continue
        i = m1.first_at_or_after(pd.Timestamp(t_in, tz="UTC"))
        fill_ref = m1.open[i] + (costs.spread if direction == "long" else 0.0)
        a = float(d["atr20"].iloc[k]) * STOP_ATR[name]
        stop = fill_ref - a if direction == "long" else fill_ref + a
        target = (fill_ref + a if direction == "long" else fill_ref - a) if name == "rsi2_revert" else \
            (np.inf if direction == "long" else -np.inf)
        kx = _rule_exit_k(name, d, k, direction)
        t_out = Clock.next_after(clock.t0930, ends[kx]) if kx is not None else None
        fd = pd.Timestamp(t_out, tz="UTC") if t_out is not None else pd.Timestamp(m1.ts[-1], tz="UTC")
        t_in_ts = pd.Timestamp(t_in, tz="UTC")
        f = simulate(m1, direction, fill_ref, stop, target, t_in_ts, t_in_ts + pd.Timedelta(minutes=1), fd,
                     costs, "expired", "rule_exit" if t_out is not None else "data_end", order_type="market")
        if not f.filled:
            continue
        trades.append(_record(name, inst, spec, direction, d.index[k], f, stop, cost_mult,
                              open_ended=name != "rsi2_revert", feats=_feats(d, k, fill_ref, a)))
        flat_from = f.exit_time.value if f.exit_time is not None else np.iinfo(np.int64).max
    return trades


def _run_overnight(inst, spec, d, m1, clock, start, end, costs, cost_mult):
    trades, flat_from = [], 0
    for t in clock.t1555:
        if t < start.value or (end is not None and t >= end.value) or t < flat_from:
            continue
        t_ts = pd.Timestamp(t, tz="UTC")
        date = t_ts.tz_convert(NY).normalize().tz_localize(None)
        # ATR of the last COMPLETE daily bar before today's close
        prev = d[d["ny_date"] < date]
        if prev.empty or not np.isfinite(prev["atr20"].iloc[-1]):
            continue
        a = float(prev["atr20"].iloc[-1])
        i = m1.first_at_or_after(t_ts)
        fill_ref = m1.open[i] + costs.spread
        t_out = Clock.next_after(clock.t0935, t)
        if t_out is None:
            break
        f = simulate(m1, "long", fill_ref, fill_ref - a, np.inf, t_ts, t_ts + pd.Timedelta(minutes=1),
                     pd.Timestamp(t_out, tz="UTC"), costs, "expired", "time_exit", order_type="market")
        if not f.filled:
            continue
        trades.append(_record("overnight", inst, spec, "long", t_ts, f, fill_ref - a, cost_mult, True,
                              _feats(prev, len(prev) - 1, fill_ref, a)))
        flat_from = f.exit_time.value
    return trades


def _feats(d: pd.DataFrame, k: int, price: float, stop_dist: float) -> dict:
    r = d.iloc[k]
    a = r["atr20"] if r["atr20"] else np.nan
    return {"f_atr_pct": float(d["atr20"].iloc[max(0, k - 250): k + 1].rank(pct=True).iloc[-1]),
            "f_ret20_atr": float((r["close"] - d["close"].iloc[k - 20]) / a) if k >= 20 else np.nan,
            "f_ret63": float(r["ret63"]) if pd.notna(r["ret63"]) else np.nan,
            "f_sma200_dist_atr": float((r["close"] - r["sma200"]) / a) if pd.notna(r["sma200"]) else np.nan,
            "f_rsi2": float(r["rsi2"]) if pd.notna(r["rsi2"]) else np.nan,
            "f_stop_atr": float(stop_dist / a) if a else np.nan, "f_weekday": int(r["weekday"])}


def _record(name, inst, spec, direction, signal_ts, f, stop, cost_mult, open_ended, feats) -> dict:
    risk = abs(f.fill_price - stop)
    nights = _nights(f.fill_time, f.exit_time)
    sw = swap_pts(spec, f.fill_price, nights, cost_mult)
    r = (f.pnl_pts - sw) / risk if risk > 0 else 0.0
    return {"strategy": name, "instrument": inst, "direction": direction, "signal_time": signal_ts,
            "fill_time": f.fill_time, "exit_time": f.exit_time, "fill_price": f.fill_price,
            "exit_price": f.exit_price, "stop": stop, "risk_pts": risk, "outcome": f.outcome, "nights": nights,
            "swap_pts": sw, "pnl_pts": f.pnl_pts, "r": r, "r_gross": f.pnl_pts / risk if risk > 0 else 0.0,
            "mfe_r": f.mfe_pts / risk if risk > 0 else 0.0, "open_ended": open_ended, **feats}


def run_all(settings: Settings, final: bool = False, cost_mult: float = 1.0, log=print,
            only: list[str] | None = None) -> pd.DataFrame:
    from src.data.freeserv_history import instrument_settings
    from src.data.store import load_bars
    h = pd.Timestamp(settings.research.holdout_start, tz="UTC")
    start, end = (h, None) if final else (pd.Timestamp(settings.backtest.start, tz="UTC"), h)
    rows = []
    for inst, spec in settings.instruments.items():
        s = instrument_settings(settings, inst)
        d1 = load_bars(s, "1d", settings.swing_history_start, end)
        # research: 1m data stops at the holdout start, so a trade still open then is closed at the
        # last research bar ("data_end") and never sees holdout prices
        m1 = M1.from_frame(load_bars(s, "1m", start - pd.Timedelta(days=5), end))
        clock = Clock.from_m1(m1)
        for name in (only or SWING_STRATEGIES):
            t = run_strategy(name, inst, spec, d1, m1, clock, start, end, cost_mult)
            log(f"  {inst:7s} {name:14s} {len(t):5d} trades")
            rows += t
    df = pd.DataFrame(rows)
    if not final and len(df):
        assert pd.DatetimeIndex(df["exit_time"]).max() <= end, "research trade used holdout data"
    return df
