"""SQLite persistence for the live engine: signals, compliance blocks, alerts/events, settings kv.

The database is the record Rald can show the prop firm: every signal, every block with its
reasons and timestamp, and every manual action. Stored at settings.live.db_path (gitignored).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  setup_key TEXT UNIQUE NOT NULL,
  created_at TEXT NOT NULL, trading_day TEXT NOT NULL,
  setup_type TEXT, direction TEXT, session TEXT,
  entry REAL, stop REAL, target REAL, rr REAL, risk_pts REAL, lots REAL, risk_usd REAL,
  expires_at TEXT, order_deadline TEXT, flat_deadline TEXT, order_reason TEXT, flat_reason TEXT,
  reason TEXT, confidence REAL, confidence_note TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  paper_outcome TEXT, paper_fill_time TEXT, paper_exit_time TEXT, paper_exit_price REAL,
  paper_r REAL, paper_pnl_usd REAL,
  action TEXT, fill_price REAL, exit_price REAL, lots_taken REAL, pnl_usd REAL, notes TEXT DEFAULT '',
  action_at TEXT
);
CREATE TABLE IF NOT EXISTS blocks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  setup_key TEXT UNIQUE NOT NULL, time TEXT NOT NULL, trading_day TEXT NOT NULL,
  setup_type TEXT, direction TEXT, reasons TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, level TEXT, text TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
"""


def iso(t) -> str | None:
    if t is None or (isinstance(t, float) and pd.isna(t)) or t is pd.NaT:
        return None
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self._lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    # ------------------------------------------------------------------ generic helpers
    def _exec(self, sql: str, args=()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def _all(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # ------------------------------------------------------------------------- signals
    def has_setup(self, key: str) -> bool:
        return bool(self._all("SELECT 1 FROM signals WHERE setup_key=? UNION ALL "
                              "SELECT 1 FROM blocks WHERE setup_key=?", (key, key)))

    def add_signal(self, row: dict) -> int:
        cols = ", ".join(row)
        q = ", ".join("?" for _ in row)
        return int(self._exec(f"INSERT INTO signals ({cols}) VALUES ({q})", tuple(row.values())).lastrowid)

    def update_signal(self, sid: int, **fields) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE signals SET {sets} WHERE id=?", (*fields.values(), sid))

    def signal(self, sid: int) -> dict | None:
        r = self._all("SELECT * FROM signals WHERE id=?", (sid,))
        return r[0] if r else None

    def signals(self, limit: int = 50, where: str = "", args=()) -> list[dict]:
        return self._all(f"SELECT * FROM signals {where} ORDER BY created_at DESC, id DESC LIMIT ?",
                         (*args, limit))

    def open_signals(self) -> list[dict]:
        return self._all("SELECT * FROM signals WHERE status IN ('active','filled') ORDER BY created_at")

    # -------------------------------------------------------------------------- blocks
    def add_block(self, key: str, time, trading_day: str, setup_type: str, direction: str,
                  reasons: list[str]) -> None:
        self._exec("INSERT OR IGNORE INTO blocks (setup_key, time, trading_day, setup_type, direction, reasons) "
                   "VALUES (?,?,?,?,?,?)", (key, iso(time), trading_day, setup_type, direction, json.dumps(reasons)))

    def blocks(self, trading_day: str | None = None) -> list[dict]:
        rows = self._all("SELECT * FROM blocks WHERE (? IS NULL OR trading_day=?) ORDER BY time DESC",
                         (trading_day, trading_day))
        for r in rows:
            r["reasons"] = json.loads(r["reasons"])
        return rows

    # -------------------------------------------------------------------------- events
    def add_event(self, time, level: str, text: str) -> None:
        self._exec("INSERT INTO events (time, level, text) VALUES (?,?,?)", (iso(time), level, text))

    def events(self, limit: int = 100) -> list[dict]:
        return self._all("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    # ------------------------------------------------------------------------------ kv
    def get(self, key: str, default=None):
        r = self._all("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(r[0]["value"]) if r else default

    def set(self, key: str, value, now=None) -> None:
        self._exec("INSERT INTO kv (key, value, updated_at) VALUES (?,?,?) "
                   "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                   (key, json.dumps(value), iso(now or pd.Timestamp.now(tz="UTC"))))
