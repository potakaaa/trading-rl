# ICT Signal Assistant (US100)

A signal-only trading assistant: detects ICT setups on US100, scores them with a
walk-forward-validated ML model, enforces prop-firm rules (Equity Edge 2-Step) and a news
filter as hard constraints, and sends signals to a dashboard, Telegram and Discord.
**It never places trades.** Full design: see [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md).

## Status
Phases 0-7 built. **Paper mode only:** the Phase 3 backtest and the one-time holdout test found
**no edge** for these rules (see `docs/LEARNING.md`, `reports/backtest_latest.html`).

## Setup (macOS)
Requires [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
uv sync                                        # Python env + dependencies
cp .env.example .env                           # FRED key; Telegram/Discord optional
uv run pytest                                  # tests

uv run python -m src.cli data download         # Dukascopy 1m history (resumable)
uv run python -m src.cli data build            # clean + resample + reports/data_quality.html
uv run python -m src.cli news build-history    # historical high-impact USD events
uv run python -m src.cli backtest run          # reports/backtest_latest.html
uv run python -m src.cli backtest research     # pre-registered rule variants (holdout locked)

uv run python -m src.cli live run              # live PAPER engine + API on :8000 (one command)
cd dashboard && npm install && npm run dev     # dashboard on http://localhost:3000
uv run python -m src.cli live paper-report     # Phase 7: paper results vs backtest
```

Values still marked UNCONFIRMED in `config/settings.yaml` (lot value, spread, news window) must be
checked against Match-Trader and Equity Edge.

## Layout
`config/` settings · `data/` parquet + live SQLite (gitignored) · `src/` Python packages ·
`dashboard/` Next.js · `reports/` generated reports (gitignored) · `tests/` · `docs/` (spec,
API contract, learning notes)

## Disclaimer
This is a research and decision-support tool, not financial advice. Backtested and paper-traded
results do not guarantee future performance. Prop firm rules change; always confirm current rules
with Equity Edge before trading.
