# Build Spec: ICT + ML Trading Signal Assistant (US100, Prop Firm Safe)

> Paste this whole file into Claude Code (or save it as `BUILD_SPEC.md` in an empty project folder and tell Claude Code: "Read BUILD_SPEC.md and start Phase 0").
> Before you start, fill in every line marked **TODO(Rald)**.

---

## 0. Your role and how to work with me

You are building a trading **signal assistant** for me (Rald). I am learning as we go, so:

- **Work in phases.** Finish one phase, show me the result, and **stop and wait for my approval** before the next phase.
- **Explain as you build.** After each phase, add a section to `docs/LEARNING.md` explaining in plain language what you built, why, and the key concepts (e.g. "what is lookahead bias", "what walk-forward testing does"). Assume I know trading and some JavaScript/Next.js, but am new to Python, data science and ML.
- **Ask before guessing** on anything that affects trading logic or prop firm rules. Never invent a rule value; use the config file and ask me.
- **Be honest about results.** If backtests show no edge, say so plainly. Do not tune parameters until the backtest looks good; that is overfitting.
- Ask me at the start: my OS (Windows/macOS), and whether Python 3.11+ and Node 20+ are installed.

**This tool never places trades.** It only produces signals. I execute manually on Match-Trader.

---

## 1. What we are building

A system that:

1. Detects ICT setups on **US100 (Nasdaq 100 CFD)** intraday.
2. Scores each setup with a machine learning model trained and validated with **walk-forward testing** on 5+ years of history.
3. Enforces **prop firm rules** (Equity Edge 2-Step) and a **live economic news filter** as hard constraints the model cannot override.
4. Sends signals (entry, stop loss, take profit, position size, confidence) to a **web dashboard**, **Telegram**, and **Discord**.
5. Runs on my laptop during the New York sessions.

---

## 2. Trading parameters

| Item | Value |
|---|---|
| Instrument | US100 CFD (Nasdaq 100 index) |
| Style | Day trading only. No positions held past session end. |
| Sessions (New York time, configurable) | NY AM 09:30–11:30, NY PM 13:30–16:00 |
| Timezone handling | All session logic in `America/New_York` (DST-aware). Display times also in `Asia/Manila`. |
| Execution | Signals only, manual execution by me |
| Prop firm | Equity Edge, **2-Step** account (static drawdown) |
| Account size | **TODO(Rald)**: e.g. $10,000 / $25,000 / $50,000 / $100,000 |
| Risk per trade | 0.5% of initial balance |
| Daily loss limit | **TODO(Rald)**: % from Equity Edge dashboard |
| Max (static) drawdown | **TODO(Rald)**: % from Equity Edge dashboard |
| News restriction window | **TODO(Rald)**: minutes before / after high-impact USD news |
| Lot value | **TODO(Rald)**: $ value per 1.0 point move for 1 lot of US100 on my Match-Trader account (check contract specs) |
| Typical spread | **TODO(Rald)**: US100 spread in points during NY session |

All of these go in `config/settings.yaml`. Nothing hard-coded.

### Internal safety buffers (stricter than the firm)
- Stop issuing signals for the day when realized daily loss reaches **50% of the daily loss limit**.
- Stop issuing signals entirely when total drawdown reaches **70% of the max drawdown**, until I manually reset.
- Max **2 signals per session**, max **3 per day** (configurable).
- After **2 consecutive losses** in a day, stop for the day.

---

## 3. Tech stack

- **Python 3.11+** for data, detection, backtesting, ML and the live engine.
  - `pandas`, `numpy`, `pyarrow` (Parquet storage), `lightgbm`, `scikit-learn`, `pydantic` (config), `fastapi` + `uvicorn` (API + WebSocket), `apscheduler`, `httpx`, `python-telegram-bot` or plain HTTP calls, `pytest`.
- **Next.js (App Router) + TypeScript + Tailwind** for the dashboard, using **TradingView Lightweight Charts** (`lightweight-charts`) for the chart.
- **SQLite** for signals, trade journal, news-block log and account state.
- Package management: `uv` or `pip` + `venv` (pick one, explain it to me).

### Data sources (pluggable via a `DataProvider` interface)
- **Historical:** Dukascopy US100 tick/1-minute data (instrument usually named `USATECHIDXUSD`). Use an existing downloader library or write one. Store as Parquet, partitioned by year/month.
- **Live:** Default to an **OANDA practice (demo) account** v20 API streaming `NAS100_USD`. It is free. Also make it possible to backfill history from OANDA as a cross-check.
- Verify each source actually works before relying on it. If one does not, tell me and propose an alternative.
- Signals must be expressed relative to structure (points from entry), because my broker's US100 price may differ slightly from the data feed. Show a "price offset" field I can set in the dashboard.

### News source (pluggable via a `NewsProvider` interface)
- **Live:** ForexFactory weekly calendar JSON feed (e.g. `nfs.faireconomy.media/ff_calendar_thisweek.json`). Refresh every 30 minutes and on startup. Filter to **USD + High impact**.
- **Historical (for backtests):** Build a dataset of past high-impact USD events (CPI, NFP, FOMC, PPI, GDP, retail sales, unemployment claims, ISM, Powell speeches, etc.) for the full backtest period. Find a reliable source and tell me which one you used. Store as Parquet.
- If the live news feed fails to load, the system must go **fail-safe**: no signals until news data is available again, and alert me.

---

## 4. Project structure

```
ict-signal-assistant/
├── config/settings.yaml
├── data/                  # raw + processed parquet (gitignored)
├── src/
│   ├── data/              # providers, downloaders, resampling, cleaning
│   ├── news/              # news providers, calendar, blackout logic
│   ├── features/          # sessions, swings, liquidity, FVG, OB, MSS, HTF bias
│   ├── setups/            # setup detectors (rule-based)
│   ├── labeling/          # outcome labels for ML
│   ├── backtest/          # engine, cost model, prop rules, metrics, reports
│   ├── ml/                # training, walk-forward, calibration, model registry
│   ├── risk/              # position sizing, prop guardrails, account state
│   ├── live/              # live engine, scheduler
│   ├── alerts/            # telegram, discord
│   └── api/               # fastapi app + websocket
├── dashboard/             # Next.js app
├── reports/               # backtest/walk-forward HTML reports
├── tests/
└── docs/LEARNING.md
```

---

## 5. ICT concept definitions (must be exact and configurable)

Every concept must be a **deterministic, unit-tested rule** using only data available at that moment (**no lookahead**). All thresholds go in config.

- **Swing high/low:** fractal pivot with N bars each side (default N=2 on 5m, N=3 on 15m). A swing is only "confirmed" N bars after it forms; use it only from the confirmation bar onward.
- **Liquidity pools:** previous day high/low, Asia session high/low, London session high/low, overnight (18:00–09:30 NY) high/low, and equal highs/lows (two+ swings within X × ATR of each other).
- **Liquidity sweep:** price trades beyond a liquidity level by at least `min_sweep` (× ATR) and closes back inside it within K bars.
- **Market structure shift (MSS):** after a sweep, a candle **closes** beyond the most recent opposing confirmed swing, with displacement: body ≥ `disp_mult` × ATR(14).
- **Fair value gap (FVG):** 3-candle pattern. Bullish: candle1.high < candle3.low. Bearish: candle1.low > candle3.high. Minimum size ≥ `min_fvg` × ATR. Track whether it has been filled.
- **Order block (OB):** the last opposite-colored candle before the displacement leg that caused the MSS. Zone = that candle's open-to-low (bullish) or open-to-high (bearish), configurable to use full range.
- **Premium/discount:** relative to the current dealing range (most recent swing high to swing low on 15m or 1h). Longs only in discount (< 50%), shorts only in premium (> 50%), toggleable.
- **Higher-timeframe bias:** from daily and 4h structure (last confirmed break of structure direction) plus price vs previous day midpoint. Output: bullish / bearish / neutral.

### Setups for v1
1. **Sweep → MSS → FVG entry**
   - Liquidity sweep inside the session window → MSS with displacement → limit entry at the FVG created in the displacement leg (entry at FVG start or 50% "consequent encroachment", configurable).
   - Stop: beyond the sweep extreme + buffer. Target: nearest opposing liquidity pool; minimum R:R 2.0 (configurable). Also log 1R, 2R, 3R outcomes.
2. **Order block retest**
   - After MSS, price returns to the OB zone in the correct premium/discount half, aligned with HTF bias.
   - Same stop/target logic.

Signals expire if not filled within `entry_expiry_bars` or before the session ends.

### Visual verification (required)
Before any backtest, build a chart tool that plots detected swings, liquidity levels, sweeps, MSS, FVGs, OBs and setups on random sample days. Generate 20 sample days for me to review. **I must confirm the detections match how I would mark the chart before we move on.**

---

## 6. Backtesting rules

- Event-driven, bar-by-bar on 1-minute data; setups detected on 5m (configurable), with 15m/1h/4h/daily context. Higher-timeframe bars only become visible after they close.
- **Costs:** spread (from config, and a 2× stress test), slippage on stops (configurable points), commission if any.
- **Intrabar ambiguity:** if SL and TP are both hit inside the same 1-minute bar, assume SL hit first (pessimistic).
- **Prop rules enforced in the backtest exactly as live:** news blackout, daily loss limit, static max drawdown, internal buffers, max trades, force-flat at session end.
- **News blackout in backtests uses the historical news dataset**, same window as live.
- Metrics: win rate, average R, expectancy (R), profit factor, max drawdown (% and R), longest losing streak, trades per month, % of days that would breach the firm's daily limit, results by setup, by session, by weekday, by year, and **probability of passing the 2-Step challenge** via Monte Carlo resampling of trades.
- Output an HTML report in `reports/` with an equity curve and all metrics.

**Gate:** if the raw rule-based setups show no positive expectancy after costs, stop and report to me before building the ML layer. ML should improve filtering, not create an edge from nothing.

---

## 7. Machine learning layer

### Approach
Meta-labeling. The rule-based detector proposes setups; the model predicts **P(TP hit before SL)** for each one. Do **not** use reinforcement learning in v1.

### Labels
For each historical setup: 1 if TP is hit before SL within the session, else 0 (using the same cost and intrabar rules as the backtest). Also store the R achieved.

### Features (computed only from data available at signal time)
- Setup type, direction, session (AM/PM), minutes since session open, weekday
- HTF bias (daily, 4h), alignment of trade with bias
- Sweep depth (× ATR), which liquidity was swept (PDH/PDL, Asia, London, equal highs/lows)
- Displacement size (× ATR), FVG size, OB size, distance from entry to target in R
- Premium/discount position (0–1)
- ATR percentile, overnight range size vs average, gap from previous close
- News context: minutes since/until nearest high-impact USD event, is-news-day flag
- Optional macro context (test it, keep only if it helps out-of-sample): VIX level/change, US10Y yield change, DXY change. Only add if a free, reliable daily source exists.

### Walk-forward validation
- Rolling or expanding windows, e.g. train 2 years → test next 3 months → roll forward 3 months. Configurable.
- **Purge/embargo:** no overlap between training labels and test periods.
- Hold out the **most recent 6–12 months as a final untouched test set**. Use it once, at the end, and report it separately.
- Model: LightGBM with conservative settings. Calibrate probabilities (isotonic or Platt) inside each training window.
- Compare in the report: (a) take all setups, (b) take setups above threshold from the model. Show performance per walk-forward fold.
- Threshold selection happens on training/validation data only, never on test data.
- Show feature importance (and SHAP if easy) so I can see what the model relies on.

### "Confidence" definition
The confidence shown on each signal is the **calibrated out-of-sample probability**, plus the historical win rate and expectancy of past setups in the same probability bucket. It must never come from in-sample results.

### Retraining
A script that retrains on new data (e.g. monthly), runs walk-forward again, and only promotes the new model if it is not worse on recent out-of-sample data. Keep versioned models in a registry with their reports.

---

## 8. Risk and prop rule engine (hard constraints)

A standalone module that every signal must pass. The ML model cannot override it.

Blocks a signal if any of these are true:
- Inside the news blackout window (from config) around any high-impact USD event, or news data unavailable
- Outside allowed sessions, or too close to session end to reach the target
- Daily loss buffer or max drawdown buffer reached
- Max trades per session/day reached, or 2 consecutive losses today
- Position size would risk more than 0.5% of initial balance
- Model confidence below threshold

Every block is logged to SQLite with timestamp and reason. The dashboard shows today's blocks. This gives me a compliance record if the firm ever questions a trade.

**Position sizing:** lots = (balance × risk%) / (stop distance in points × lot value per point), rounded **down** to the broker's lot step.

**Account state:** since execution is manual, the dashboard lets me mark each signal as taken / skipped and enter the actual fill and exit. Daily P&L and drawdown are tracked from these entries. Also allow a manual "current balance" override.

---

## 9. Live engine

- Starts with one command. Streams live US100 data, builds candles, runs the same detection code as the backtest (shared code, not a copy).
- Refreshes news on schedule; shows a countdown to the next high-impact event.
- When a setup passes all checks, it creates a signal: direction, entry, SL, TP, R:R, lot size, confidence, setup type, reason summary, and expiry time.
- Sends the signal to the dashboard (WebSocket), Telegram and Discord simultaneously.
- Also sends: "news blackout starting/ending", "daily stop reached", "data feed disconnected", "engine started/stopped".
- Heartbeat: if data stops for more than 60 seconds during a session, alert me.
- Keep a startup checklist that warns if my laptop sleep settings might stop it (explain how to disable sleep on my OS).

### Alerts setup
- Telegram: bot token and chat ID in `.env`. Explain step by step how to create the bot with BotFather.
- Discord: webhook URL in `.env`. Explain how to create it.
- Never commit `.env`. Provide `.env.example`.

---

## 10. Dashboard (Next.js + Lightweight Charts)

Dark theme, clean and minimal.

- **Main chart:** US100 candles (1m/5m/15m toggle), with overlays for liquidity levels, FVGs, OBs, sweeps, MSS, and signal markers with entry/SL/TP lines.
- **Signal panel:** active and recent signals with confidence, setup type, reasoning, lot size, expiry. Buttons: Taken / Skipped, then a form for fill and exit.
- **Risk panel:** today's P&L, daily loss used vs limit (progress bar), max drawdown used vs limit, trades taken today, consecutive losses, status (ACTIVE / PAUSED-NEWS / STOPPED-FOR-DAY).
- **News panel:** today's high-impact USD events in NY and Manila time, countdown, blackout status.
- **Backtest page:** view the latest walk-forward report and model version.
- **Journal page:** all signals and outcomes, filterable, with stats.
- Settings page to view config values (read-only is fine for v1).

---

## 11. Phases (stop after each one for my review)

**Phase 0: Setup**
Repo, environment, config file with TODO values, README with how to run, `docs/LEARNING.md`. Ask me for my OS and the TODO values.

**Phase 1: Data**
Historical downloader (5+ years US100 1-minute), cleaning (gaps, weekends, bad ticks), resampling to 5m/15m/1h/4h/daily, data quality report. Live data provider connection test. Historical news dataset + live news feed test.

**Phase 2: ICT detectors + visual check**
All concepts in section 5 with unit tests. Generate 20 annotated sample charts. **Wait for my confirmation.**

**Phase 3: Backtester + prop rules**
Engine, costs, prop/news enforcement, metrics, HTML report, Monte Carlo challenge pass rate. Report raw-setup results honestly. **Gate check.**

**Phase 4: ML + walk-forward**
Labels, features, walk-forward training, calibration, final holdout test, comparison report.

**Phase 5: Live engine + alerts**
Live streaming, signal generation, risk engine, Telegram + Discord, logging, heartbeat.

**Phase 6: Dashboard**
Everything in section 10, connected to the API.

**Phase 7: Paper trading mode**
Run live for at least 4 weeks, logging every signal. Weekly comparison of live results vs backtest expectations. Flag if live performance drifts outside the expected range.

---

## 12. Non-negotiable rules for you (Claude Code)

1. No lookahead bias anywhere. Write tests that check features at time T only use data ≤ T.
2. Backtest and live use the same detection and risk code.
3. Never tune on the final holdout set.
4. Never weaken a prop rule or safety buffer without my explicit approval.
5. Fail safe: missing data, missing news or errors mean **no signals**, plus an alert.
6. Keep everything configurable in `settings.yaml`.
7. Write tests for detectors, risk engine, position sizing, news blackout and timezone/DST handling (including the November and March DST switches).
8. After each phase: summarize what was built, show results, update `docs/LEARNING.md`, and list what you need from me next.

---

## 13. Disclaimer to keep in the README

This is a research and decision-support tool, not financial advice. Backtested and paper-traded results do not guarantee future performance. Prop firm rules change; always confirm current rules with Equity Edge before trading.
