"""Round 8: NQ futures trades (with aggressor side) from Databento, ONLY in the pre-registered windows,
with a hard spending cap. Every day is saved once (data/databento/nq_trades/YYYY-MM-DD.parquet), so a
re-run never pays twice; spend is metered and stored in data/databento/ledger.json."""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

NY = "America/New_York"
WINDOWS = [("09:57", "10:00"), ("10:27", "10:30")]          # amended (budget), docs/STRATEGIES_R8.md
BUDGET_USD = 75.0
USD_PER_RECORD = 0.1494 / 68020      # MEASURED: Databento's official quote for 68,020 trade records (2024-03-13)
SAFETY = 1.10                        # meter 10% above the measured official rate
ROOT = PROJECT_ROOT / "data" / "databento"
DAY_DIR = ROOT / "nq_trades"
LEDGER = ROOT / "ledger.json"


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path = LEDGER, budget: float = BUDGET_USD):
        self.path, self.budget = path, budget
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(path.read_text()) if path.exists() else {"spent_usd": 0.0, "records": 0, "days": 0}

    @property
    def spent(self) -> float:
        return self.data["spent_usd"]

    def charge(self, records: int) -> None:
        with self.lock:
            self.data["spent_usd"] += records * USD_PER_RECORD * SAFETY
            self.data["records"] += records
            self.data["days"] += 1
            self.path.write_text(json.dumps(self.data, indent=1))

    def check(self, next_estimate: float = 0.0) -> None:
        if self.spent + next_estimate > self.budget:
            raise BudgetExceeded(f"budget cap: spent ${self.spent:.2f} + next ~${next_estimate:.2f} "
                                 f"> ${self.budget:.2f}")


def client():
    import databento as db
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY missing in .env")
    return db.Historical(key)


def _utc(day: str, hhmm: str) -> str:
    return pd.Timestamp(f"{day} {hhmm}", tz=NY).tz_convert("UTC").isoformat()


def fetch_day(c, day: str, retries: int = 4) -> pd.DataFrame:
    parts = []
    for a, b in WINDOWS:
        for k in range(retries):
            try:
                store = c.timeseries.get_range(dataset="GLBX.MDP3", symbols=["NQ.v.0"], stype_in="continuous",
                                               schema="trades", start=_utc(day, a), end=_utc(day, b))
                df = store.to_df()
                break
            except Exception as e:  # noqa: BLE001
                if "budget" in str(e).lower() or "credit" in str(e).lower() or k == retries - 1:
                    raise
                time.sleep(5 * (k + 1))
        if len(df):
            parts.append(df[["price", "size", "side"]])
    return pd.concat(parts) if parts else pd.DataFrame(columns=["price", "size", "side"])


def download(days: list[str], workers: int = 4, log=print) -> dict:
    DAY_DIR.mkdir(parents=True, exist_ok=True)
    ledger = Ledger()
    todo = [d for d in days if not (DAY_DIR / f"{d}.parquet").exists()]
    log(f"{len(days)} days requested, {len(todo)} to download, spent so far ${ledger.spent:.2f}")
    c = client()
    est_day = 0.06                                    # conservative per-day estimate for the cap check

    def one(day):
        ledger.check(est_day)
        df = fetch_day(c, day)
        df.to_parquet(DAY_DIR / f"{day}.parquet")
        ledger.charge(len(df))
        return day, len(df)

    done = 0
    try:
        with ThreadPoolExecutor(workers) as ex:
            for day, n in ex.map(one, todo):
                done += 1
                if done % 50 == 0:
                    log(f"  {day}: {done}/{len(todo)} days, spent ${ledger.spent:.2f}")
    except BudgetExceeded as e:
        log(f"STOPPED: {e}")
    return {"downloaded": done, "spent_usd": ledger.spent, "records": ledger.data["records"]}


def load_day(day: str) -> pd.DataFrame | None:
    p = DAY_DIR / f"{day}.parquet"
    return pd.read_parquet(p) if p.exists() else None
