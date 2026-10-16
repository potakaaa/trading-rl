"""Startup checklist: things that would silently stop signals if nobody looked."""
from __future__ import annotations

import os
import platform
import re
import subprocess

from src.config import PROJECT_ROOT, Settings


def mac_sleep_minutes() -> int | None:
    """System sleep timer from `pmset -g` (0 = never sleeps), None if unknown."""
    try:
        out = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"^\s*sleep\s+(\d+)", out, re.MULTILINE)
    return int(m.group(1)) if m else None


def run_checklist(settings: Settings) -> list[tuple[str, bool, str]]:
    """(item, ok, advice) rows."""
    rows: list[tuple[str, bool, str]] = []
    missing = settings.missing_todos()
    rows.append(("settings complete", not missing, "fill: " + ", ".join(missing) if missing else ""))
    if settings.broker.unconfirmed:
        rows.append(("broker values confirmed", False,
                     "UNCONFIRMED: " + ", ".join(settings.broker.unconfirmed)
                     + " (check the Match-Trader order ticket)"))
    rows.append(("mode", settings.live.mode == "paper", f"mode = {settings.live.mode}"))
    env = (PROJECT_ROOT / ".env").exists()
    tg = bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))
    dc = bool(os.environ.get("DISCORD_WEBHOOK_URL"))
    rows.append((".env present", env, "" if env else "cp .env.example .env"))
    rows.append(("Telegram alerts", tg, "" if tg else "see docs/LEARNING.md -> 'Telegram bot setup'"))
    rows.append(("Discord alerts", dc, "" if dc else "see docs/LEARNING.md -> 'Discord webhook setup'"))
    if platform.system() == "Darwin":
        sleep = mac_sleep_minutes()
        ok = sleep == 0
        rows.append(("Mac will not sleep", ok,
                     "" if ok else f"system sleep after {sleep} min. The engine runs under `caffeinate -i` "
                                   "(prevents idle sleep while it runs) but CLOSING THE LID still sleeps the Mac. "
                                   "Keep it plugged in and open, or System Settings > Battery > Options > "
                                   "'Prevent automatic sleeping when the display is off'."))
    return rows


def format_checklist(rows) -> str:
    return "\n".join(f"  [{'OK' if ok else '!!'}] {item}" + (f" - {adv}" if adv else "") for item, ok, adv in rows)
