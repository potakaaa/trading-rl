"""Round-4 dataset (docs/STRATEGIES_R4.md): one row per decision time with features known at that time
and the net-R labels of the fixed long and short trades (simulated with the backtest's fill code)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.fills import M1, Costs, simulate
from src.config import Settings
from src.features.basics import atr
from src.strategies.rules import DailyContext
from src.timeutils import NY

DECISION_TIMES = ["09:45", "10:00", "10:15", "10:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00"]
HORIZON_MIN = 60
STOP_ATR, TARGET_R = 1.0, 1.5
FEATURES = ["ret5", "ret15", "ret30", "ret60", "ret120", "d_open", "d_vwap", "d_pdh", "d_pdl", "d_onh", "d_onl",
            "rv30", "rv60", "atr15_atrd", "day_range_atrd", "vol_ratio30", "minute", "weekday", "gap_atrd",
            "prev_ret_atrd", "news_to_min", "news_since_min"]
MIN_NS = 60_000_000_000


def _session_end(t: pd.Timestamp) -> pd.Timestamp:
    ny = t.tz_convert(NY)
    end = "11:30" if ny.hour < 12 else "16:00"
    h, m = map(int, end.split(":"))
    return ny.normalize().replace(hour=h, minute=m).tz_convert("UTC")


def build(m1df: pd.DataFrame, b15: pd.DataFrame, d1: pd.DataFrame, news: pd.DataFrame | None,
          settings: Settings, cost_mults=(1.0, 2.0), log=print) -> pd.DataFrame:
    m1df = m1df.sort_index()
    M = M1.from_frame(m1df)
    ts, o, h, lo, c = M.ts, M.open, M.high, M.low, M.close
    vol = m1df["volume"].to_numpy(dtype=float)
    ny = pd.DatetimeIndex(ts, tz="UTC").tz_convert(NY)
    mins = (ny.hour * 60 + ny.minute).to_numpy()
    dates = ny.normalize().tz_localize(None).to_numpy()

    b15 = b15.sort_index().copy()
    b15["atr"] = atr(b15, 14)
    a15_end = pd.DatetimeIndex(b15["end"]).as_unit("ns").asi8
    a15 = b15["atr"].to_numpy()
    dctx = DailyContext(d1)
    ev = np.sort(pd.DatetimeIndex(news["ts"]).as_unit("ns").asi8) if news is not None and len(news) else None

    # per-day helpers: 09:30 open, VWAP pieces, previous session close (15:59 bar)
    uniq = np.unique(dates)
    day_first = {d: i for d, i in zip(*np.unique(dates, return_index=True), strict=False)}
    prev_close: dict = {}
    prev_open: dict = {}
    last_close = last_open = None
    for d in uniq:
        i0 = day_first[d]
        j = np.searchsorted(dates, d, side="right")
        sl = slice(i0, j)
        m = mins[sl]
        k930 = np.flatnonzero(m == 570)
        k1559 = np.flatnonzero(m == 959)
        prev_close[d], prev_open[d] = last_close, last_open
        if len(k1559) and len(k930):
            last_close, last_open = c[i0 + k1559[0]], o[i0 + k930[0]]

    # tick volume of the last 30 min at each decision slot, for the 20-day relative volume feature
    rows = []
    slot_hist: dict[str, list[float]] = {s: [] for s in DECISION_TIMES}
    for d in uniq:
        i0 = day_first[d]
        j = int(np.searchsorted(dates, d, side="right"))
        m = mins[i0:j]
        k930 = np.flatnonzero(m == 570)
        if not len(k930):
            continue
        s0 = i0 + k930[0]
        open930 = o[s0]
        tp = (h[s0:j] + lo[s0:j] + c[s0:j]) / 3
        vv = np.where(vol[s0:j] > 0, vol[s0:j], 0.0)
        cum_pv, cum_v = np.cumsum(tp * vv), np.cumsum(vv)
        date = pd.Timestamp(d)
        ctx = dctx.get(date)
        # overnight window 18:00 (prev day) .. 09:30
        on_lo_i = int(np.searchsorted(ts, (date - pd.Timedelta(hours=6)).tz_localize(NY).value))  # 18:00 prev
        on = slice(on_lo_i, s0)
        onh, onl = (h[on].max(), lo[on].min()) if s0 > on_lo_i else (np.nan, np.nan)
        for slot in DECISION_TIMES:
            hh, mm = map(int, slot.split(":"))
            t_ns = (date + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(NY).value
            i = int(np.searchsorted(ts, t_ns))                 # entry bar (first bar starting >= t)
            if i >= len(ts) or ts[i] - t_ns > 2 * MIN_NS or i - 1 < s0:
                continue
            k = i - 1                                          # last COMPLETE 1m bar at t
            q = int(np.searchsorted(a15_end, t_ns, side="right")) - 1
            if q < 14 or not np.isfinite(a15[q]) or a15[q] <= 0:
                continue
            A = a15[q]
            ct = c[k]

            def back(mn, k=k, t_ns=t_ns):
                p = int(np.searchsorted(ts, t_ns - mn * MIN_NS)) - 1
                return c[p] if p >= 0 else np.nan

            seg30 = slice(max(int(np.searchsorted(ts, t_ns - 30 * MIN_NS)), 0), k + 1)
            seg60 = slice(max(int(np.searchsorted(ts, t_ns - 60 * MIN_NS)), 0), k + 1)
            vw_i = k - s0
            vwap = cum_pv[vw_i] / cum_v[vw_i] if cum_v[vw_i] > 0 else np.nan
            v30 = float(vol[seg30].sum())
            hist = slot_hist[slot]
            vol_ratio = v30 / np.mean(hist[-20:]) if len(hist) >= 5 and np.mean(hist[-20:]) > 0 else np.nan
            hist.append(v30)
            atrd = ctx["atr_d"] if ctx else np.nan
            pc, po = prev_close.get(d), prev_open.get(d)
            f = {
                "t": pd.Timestamp(t_ns, tz="UTC"), "date": date, "slot": slot, "atr15": A, "price": ct,
                "ret5": (ct - back(5)) / A, "ret15": (ct - back(15)) / A, "ret30": (ct - back(30)) / A,
                "ret60": (ct - back(60)) / A, "ret120": (ct - back(120)) / A,
                "d_open": (ct - open930) / A, "d_vwap": (ct - vwap) / A,
                "d_pdh": (ct - ctx["pdh"]) / A if ctx else np.nan, "d_pdl": (ct - ctx["pdl"]) / A if ctx else np.nan,
                "d_onh": (ct - onh) / A, "d_onl": (ct - onl) / A,
                "rv30": np.std(np.diff(c[seg30])) * np.sqrt(30) / A if seg30.stop - seg30.start > 3 else np.nan,
                "rv60": np.std(np.diff(c[seg60])) * np.sqrt(60) / A if seg60.stop - seg60.start > 3 else np.nan,
                "atr15_atrd": A / atrd if atrd else np.nan,
                "day_range_atrd": (h[s0:k + 1].max() - lo[s0:k + 1].min()) / atrd if atrd else np.nan,
                "vol_ratio30": vol_ratio, "minute": hh * 60 + mm, "weekday": date.weekday(),
                "gap_atrd": (open930 - pc) / atrd if pc is not None and atrd else np.nan,
                "prev_ret_atrd": (pc - po) / atrd if pc is not None and po is not None and atrd else np.nan,
            }
            if ev is not None:
                jn = int(np.searchsorted(ev, t_ns))
                f["news_to_min"] = min((ev[jn] - t_ns) / MIN_NS, 720) if jn < len(ev) else 720.0
                f["news_since_min"] = min((t_ns - ev[jn - 1]) / MIN_NS, 720) if jn > 0 else 720.0
            else:
                f["news_to_min"] = f["news_since_min"] = 720.0
            # ---- labels (future; never used as features)
            t_ts = pd.Timestamp(t_ns, tz="UTC")
            fd = min(t_ts + pd.Timedelta(minutes=HORIZON_MIN), _session_end(t_ts))
            fut = slice(i, int(np.searchsorted(ts, fd.value)))
            f["range60"] = (h[fut].max() - lo[fut].min()) / A if fut.stop > fut.start else np.nan
            for mult in cost_mults:
                sp, sl = settings.broker.typical_spread_points * mult, settings.backtest.stop_slippage_points * mult
                costs = Costs(sp, sl)
                for side in ("long", "short"):
                    fill = o[i] + (sp if side == "long" else 0.0)
                    stop = fill - STOP_ATR * A if side == "long" else fill + STOP_ATR * A
                    tgt = fill + TARGET_R * STOP_ATR * A if side == "long" else fill - TARGET_R * STOP_ATR * A
                    fl = simulate(M, side, fill, stop, tgt, t_ts, t_ts + pd.Timedelta(minutes=1), fd, costs,
                                  "expired", "time_exit", order_type="market")
                    key = f"r_{side}" + ("" if mult == 1.0 else f"_{mult:g}x")
                    f[key] = fl.pnl_pts / (STOP_ATR * A) if fl.filled else np.nan
                    if mult == 1.0:
                        f[f"exit_{side}"] = fl.exit_time
            rows.append(f)
    df = pd.DataFrame(rows)
    log(f"  {len(df)} decisions over {df['date'].nunique()} days")
    return df


def load_dataset(settings: Settings, start: str, end: str, log=print) -> pd.DataFrame:
    from src.data.store import load_bars
    from src.news.historical import load_history
    s0 = pd.Timestamp(start, tz="UTC")
    m1 = load_bars(settings, "1m", s0 - pd.Timedelta(days=3), end)
    b15 = load_bars(settings, "15m", s0 - pd.Timedelta(days=10), end)
    d1 = load_bars(settings, "1d", s0 - pd.Timedelta(days=60), end)
    df = build(m1, b15, d1, load_history(settings), settings, log=log)
    return df[(df["t"] >= s0) & (df["t"] < pd.Timestamp(end, tz="UTC"))].reset_index(drop=True)
