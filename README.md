# ICT Signal Assistant (US100)

A signal-only trading assistant: detects ICT setups on US100, scores them with a
walk-forward-validated ML model, enforces prop-firm rules (Equity Edge 2-Step) and a news
filter as hard constraints, and sends signals to a dashboard, Telegram and Discord.
**It never places trades.** Full design: see [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md).

## Status
Phase 0 (setup) complete. See `docs/LEARNING.md`.

## Setup (macOS)
Requires [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
uv sync                      # creates .venv and installs dependencies
cp .env.example .env         # fill in later phases (OANDA, Telegram, Discord)
uv run pytest                # run tests
```

Fill in every `null` value marked `TODO(Rald)` in `config/settings.yaml` before Phase 3.

## Layout
`config/` settings · `data/` parquet (gitignored) · `src/` Python packages · `dashboard/` Next.js (Phase 6) ·
`reports/` backtest reports · `tests/` · `docs/LEARNING.md`

## Disclaimer
This is a research and decision-support tool, not financial advice. Backtested and paper-traded
results do not guarantee future performance. Prop firm rules change; always confirm current rules
with Equity Edge before trading.
