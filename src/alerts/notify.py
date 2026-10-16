"""Alert fan-out: Telegram + Discord (+ always the local log). Secrets come from .env only.

TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  -> https://api.telegram.org/bot<token>/sendMessage
DISCORD_WEBHOOK_URL                     -> POST {"content": ...}
A channel without credentials is skipped silently (logged once). Sending never raises into the
engine: a failed alert is logged, the engine keeps its fail-safe behaviour.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable

import httpx

log = logging.getLogger("alerts")


class Notifier:
    def __init__(self, env: dict | None = None, transport: httpx.BaseTransport | None = None,
                 sink: Callable[[str, str], None] | None = None):
        env = os.environ if env is None else env
        self.tg_token = env.get("TELEGRAM_BOT_TOKEN") or None
        self.tg_chat = env.get("TELEGRAM_CHAT_ID") or None
        self.discord = env.get("DISCORD_WEBHOOK_URL") or None
        self._transport = transport
        self.sink = sink                      # e.g. push to the dashboard websocket
        self.sent: list[tuple[str, str]] = []  # (channel, text) for tests / status
        self.errors: list[str] = []

    @property
    def channels(self) -> list[str]:
        return [c for c, ok in (("telegram", self.tg_token and self.tg_chat), ("discord", self.discord)) if ok]

    def send(self, text: str, level: str = "info") -> None:
        prefix = {"critical": "🚨 ", "warning": "⚠️ ", "signal": "📈 "}.get(level, "")
        msg = prefix + text
        log.info("[%s] %s", level, text)
        if self.sink:
            try:
                self.sink(level, text)
            except Exception as e:  # noqa: BLE001 - a UI failure must never stop alerts
                log.warning("alert sink failed: %s", e)
        with httpx.Client(timeout=10, transport=self._transport) as c:
            if self.tg_token and self.tg_chat:
                self._post(c, "telegram", f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                           {"chat_id": self.tg_chat, "text": msg, "disable_web_page_preview": True})
            if self.discord:
                self._post(c, "discord", self.discord, {"content": msg[:1900]})

    def _post(self, c: httpx.Client, channel: str, url: str, payload: dict) -> None:
        try:
            r = c.post(url, json=payload)
            if r.status_code >= 300:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            self.sent.append((channel, payload.get("text") or payload.get("content")))
        except Exception as e:  # noqa: BLE001
            err = f"{channel} alert failed: {e}"
            self.errors.append(err)
            log.warning(err)


def format_signal(sig: dict, offset: float = 0.0) -> str:
    d = "SHORT" if sig["direction"] == "short" else "LONG"

    def p(x):
        return f"{x + offset:.1f}"

    lines = [
        f"PAPER SIGNAL (unproven) — US100 {d} · {sig['setup_type']} · {sig['session']}",
        f"Entry {p(sig['entry'])} | SL {p(sig['stop'])} | TP {p(sig['target'])}  (R:R {sig['rr']:.2f})",
        f"Stop {sig['risk_pts']:.1f} pts · {sig['lots']:.2f} lots · risk ${sig['risk_usd']:.2f}",
        f"Expires {sig['expires_at']} UTC · {sig['confidence_note']}",
        f"Why: {sig['reason']}",
    ]
    if offset:
        lines.append(f"(broker prices = feed {offset:+.1f})")
    return "\n".join(lines)
