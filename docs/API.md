# Live API contract (FastAPI backend ⇄ Next.js dashboard)

Base URL: `http://127.0.0.1:8000` (configurable `live.api_port`). All times are ISO-8601 UTC strings
(`2026-10-16T13:45:00Z`); the dashboard converts them to New York and Manila time. Prices are in index
points (feed prices). Every signal is **PAPER ONLY / UNPROVEN** until Rald decides otherwise
(`mode` field). CORS allows `http://localhost:3000`.

## REST

### `GET /api/status`
```json
{
  "mode": "paper",                       // paper | live (live is never set automatically)
  "engine": {"running": true, "started_at": "...", "last_bar_at": "...", "feed_ok": true,
             "feed_lag_seconds": 18.0, "feed_error": null},
  "status": "ACTIVE",                    // ACTIVE | PAUSED-NEWS | STOPPED-FOR-DAY | HALTED | NO-DATA | NO-NEWS | OUTSIDE-SESSION
  "status_reason": "",
  "session": "ny_am",                    // or null
  "price": 29875.5, "price_offset": 0.0,
  "risk": {"initial_balance": 2500, "balance": 2487.5, "day_pnl": -12.5,
           "daily_limit_usd": 100, "daily_buffer_usd": 50, "daily_loss_used_usd": 12.5,
           "max_dd_usd": 200, "dd_buffer_usd": 140, "drawdown_used_usd": 12.5,
           "signals_today": 1, "max_signals_per_day": 3, "consecutive_losses": 1,
           "max_consecutive_losses": 2, "halted": false, "halt_reason": ""},
  "news": {"available": true, "blackout": false, "reason": "", "next_event": {"title": "CPI m/m", "ts": "..."},
           "minutes_to_next": 95.0, "last_refresh": "..."},
  "warnings": ["lot_value_per_point is UNCONFIRMED", "..."]
}
```

### `GET /api/bars?tf=1m|5m|15m&limit=500`
`{"tf": "5m", "bars": [{"t": "...", "o": 1, "h": 1, "l": 1, "c": 1}]}` (t = bar start, ascending)

### `GET /api/overlays?day=YYYY-MM-DD` (default: current trading day)
```json
{"levels": [{"kind": "PDH", "price": 1, "side": "high", "known_at": "...", "consumed_at": null}],
 "fvgs":   [{"direction": "bull", "top": 1, "bottom": 1, "start": "...", "end": "...", "filled_at": null}],
 "order_blocks": [{"top": 1, "bottom": 1, "start": "...", "direction": "bear"}],
 "sweeps": [{"t": "...", "price": 1, "level_kind": "ON_H", "direction": "short"}],
 "mss":    [{"t": "...", "price": 1, "direction": "short"}],
 "swings": [{"t": "...", "price": 1, "kind": "high"}]}
```

### `GET /api/signals?limit=50`
`{"signals": [Signal]}` newest first.
```json
Signal = {"id": 12, "created_at": "...", "setup_type": "sweep_mss_fvg", "direction": "short",
  "session": "ny_am", "entry": 1, "stop": 1, "target": 1, "rr": 2.4, "risk_pts": 20.1,
  "entry_broker": 1, "stop_broker": 1, "target_broker": 1,      // feed price + price_offset
  "lots": 0.62, "risk_usd": 12.46, "expires_at": "...",
  "confidence": null, "confidence_note": "no model (Phase 4 skipped); setup-type backtest win rate 28%",
  "reason": "ON_H swept 0.31xATR, MSS 1.4xATR, FVG CE entry, bias bearish",
  "status": "active",              // active | expired | filled | closed | cancelled
  "paper": {"outcome": null, "fill_time": null, "exit_time": null, "r": null},
  "action": null,                  // null | taken | skipped  (Rald's manual decision)
  "fill_price": null, "exit_price": null, "pnl_usd": null, "notes": ""}
```

### `POST /api/signals/{id}/action`
Body `{"action": "taken" | "skipped", "fill_price": 1.0?, "exit_price": 1.0?, "lots": 0.5?, "notes": ""?}`
→ the updated `Signal`. `pnl_usd` is computed when fill and exit are present; the account state is
rebuilt from these entries.

### `GET /api/blocks?day=YYYY-MM-DD` → `{"blocks": [{"time": "...", "setup_type": "...", "direction": "...", "reasons": ["news_blackout"]}]}`

### `GET /api/news` → `{"available": true, "last_refresh": "...", "events": [{"ts": "...", "title": "...", "blackout_start": "...", "blackout_end": "..."}]}` (this week, USD high impact)

### `GET /api/journal?from=&to=&setup_type=&action=` → `{"signals": [Signal], "stats": {"taken": 3, "wins": 1, "losses": 2, "total_r": -0.5, "total_pnl_usd": -6.2, "paper_total_r": 0.4, "paper_trades": 5}}`

### `GET /api/backtest` → `{"report_url": "/reports/backtest_latest.html", "research_url": "/reports/research_latest.html", "summary": {...raw and guarded summaries...}, "gate": {"pass": false, "text": "..."}, "model_version": null}`
`/reports/*` is served statically by the backend.

### `GET /api/settings` → the full settings.yaml as JSON (read-only), with `.env` secrets never included.

### `POST /api/account/balance` `{"balance": 2490.0}` · `POST /api/account/reset` `{}` (clears a drawdown halt) · `POST /api/price-offset` `{"offset": -1.5}`

## WebSocket `ws://127.0.0.1:8000/ws`
Server pushes JSON messages `{"type": ..., "data": ...}`:
- `status` – same payload as `GET /api/status` (on change and every 15 s)
- `bar` – `{"tf": "1m", "bar": {...}}` for every new completed 1m bar (dashboard aggregates 5m/15m itself or refetches)
- `signal` – a new or updated `Signal`
- `block` – a new block log entry
- `alert` – `{"level": "info|warning|critical", "text": "..."}` (news blackout start/end, daily stop, feed down, engine start/stop)
