"""Trade sequencing shared by the round-4 GBM and round-6 CNN (no ML library imports here, so the
CNN process never loads LightGBM: on macOS LightGBM and PyTorch must not share a process)."""
from __future__ import annotations

import pandas as pd

MAX_PER_SESSION, MAX_PER_DAY = 2, 3


def sequence(df: pd.DataFrame, choose) -> pd.DataFrame:
    """Walk decisions in time order with one position at a time and the session/day caps.
    `choose(row) -> 'long' | 'short' | None`. Returns the trades taken."""
    rows = []
    for _, day in df.sort_values("t").groupby("date", sort=True):
        busy_until = None
        per_sess = {"am": 0, "pm": 0}
        n_day = 0
        for _, r in day.iterrows():
            if busy_until is not None and r["t"] < busy_until:
                continue
            sess = "am" if r["minute"] < 12 * 60 else "pm"
            if per_sess[sess] >= MAX_PER_SESSION or n_day >= MAX_PER_DAY:
                continue
            side = choose(r)
            if side is None or pd.isna(r[f"r_{side}"]):
                continue
            rows.append({"t": r["t"], "date": r["date"], "side": side, "r": r[f"r_{side}"],
                         "r_2x": r[f"r_{side}_2x"]})
            busy_until = pd.Timestamp(r[f"exit_{side}"]) if pd.notna(r[f"exit_{side}"]) else r["t"]
            per_sess[sess] += 1
            n_day += 1
    return pd.DataFrame(rows, columns=["t", "date", "side", "r", "r_2x"])
