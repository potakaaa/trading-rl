# ICT Signal Assistant (US100)

A signal-only trading assistant: detects ICT setups on US100, scores them with a
walk-forward-validated ML model, enforces prop-firm rules (Equity Edge 2-Step) and a news
filter as hard constraints, and sends signals to a dashboard, Telegram and Discord.
**It never places trades.** Full design: see [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md).

## Status

| Phase | State |
|---|---|
| 0 Setup | done |
| 1 Data (history, cleaning, resampling, quality report, live feed, news) | code done; **run the downloads on your Mac** (below) |
| 2 ICT detectors + visual check | code + tests done; **waiting for your review of 20 real sample days** |
| 3 Backtester + prop rules | not started (needs the remaining `TODO(Rald)` values) |

Plain-language explanations of everything are in [`docs/LEARNING.md`](docs/LEARNING.md).

## Setup (macOS)
Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`) and, from Phase 6, Node 20+.

```bash
uv sync                      # creates .venv and installs dependencies
cp .env.example .env         # then fill in FRED_API_KEY and the OANDA values
uv run pytest                # run the tests (all should pass)
```

## Phase 1 + 2: run these on your Mac, in order

```bash
# 1. Historical 1-minute US100 data from Dukascopy (2019 -> yesterday).
#    ~2,800 small files; resumable: if it stops or reports failed days, just run it again.
uv run python -m src.cli data download

# 2. Clean, resample to 5m/15m/1h/4h/1d, and write the data quality report.
uv run python -m src.cli data build
open reports/data_quality.html

# 3. Live economic calendar (ForexFactory feed) - prints this week's USD high-impact events.
uv run python -m src.cli news check-live

# 4. Historical news dataset for backtests (needs FRED_API_KEY in .env).
uv run python -m src.cli news build-history

# 5. Live data connection test (needs OANDA_API_TOKEN + OANDA_ACCOUNT_ID in .env).
uv run python -m src.cli data check-live
uv run python -m src.cli data cross-check --days 5   # Dukascopy vs OANDA on the same minutes

# 6. Phase 2 visual check: 20 annotated sample days.
uv run python -m src.cli charts sample --n 20
open reports/samples/index.html
```

Send me the console output of each step (or the error), plus your notes on the sample charts.

## Layout
`config/settings.yaml` all settings · `src/data` history, cleaning, resampling, live providers ·
`src/news` calendar feeds + blackout · `src/features` ICT concepts · `src/setups` setup rules ·
`src/viz` charts · `src/cli.py` commands · `data/` parquet (gitignored) · `reports/` generated HTML ·
`tests/` · `docs/LEARNING.md`

Fill in every `null` value marked `TODO(Rald)` in `config/settings.yaml` before Phase 3.

## Disclaimer
This is a research and decision-support tool, not financial advice. Backtested and paper-traded
results do not guarantee future performance. Prop firm rules change; always confirm current rules
with Equity Edge before trading.
