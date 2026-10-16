# Dashboard (Phase 6)

Next.js (App Router) + TypeScript + Tailwind dashboard for the US100 signal assistant. It reads the
FastAPI live backend described in [`../docs/API.md`](../docs/API.md).

> **PAPER MODE.** Every signal is unproven. The backtest showed no edge. This is not financial advice.

## Run

```bash
# 1. Start the backend (from the repo root)
uv run python -m src.cli live run

# 2. Start the dashboard
cd dashboard
npm install
npm run dev          # http://localhost:3000
```

Production build: `npm run build && npm start`. Lint: `npm run lint`.

## Configuration

| Env var                | Default                 | Purpose                                          |
| ---------------------- | ----------------------- | ------------------------------------------------ |
| `NEXT_PUBLIC_API_BASE` | `http://127.0.0.1:8000` | Backend REST base. The WebSocket is `<base>/ws` (http→ws). |

Copy `.env.example` to `.env.local` to override it. The backend's CORS allows `http://localhost:3000`,
so open the dashboard on that origin. `NEXT_PUBLIC_*` values are baked in at build time.

## Pages

- `/`: status bar and paper banner, the US100 chart (1m/5m/15m, liquidity levels, FVG/OB zones,
  sweep/MSS/swing markers, signal entry/SL/TP lines), signals with Taken/Skipped plus a fill form, the risk
  panel (limits, reset halt, balance and price-offset overrides), news in NY and Manila time, today's
  blocks, and live alerts.
- `/backtest`: gate verdict, summary, and embedded backtest and research reports.
- `/journal`: filterable signal history with stats.
- `/settings`: read-only view of `settings.yaml`.

## Layout

- `src/lib/api.ts`: typed API client. Its types mirror `docs/API.md`.
- `src/lib/hooks.ts`: polling fetch hook, ticking clock, and a WebSocket hook with backoff reconnect.
- `src/lib/bars.ts`: client-side aggregation of live 1m bars into 5m and 15m.
- `src/components/CandleChart.tsx`: lightweight-charts v5 chart, loaded on the client only.

If the backend is down, every page shows a "Backend offline" notice and keeps retrying.
