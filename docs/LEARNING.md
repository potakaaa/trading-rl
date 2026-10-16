# Learning notes

## Phase 0: Setup

**What was built:** project skeleton, a Python environment, a config file, and a config loader.

**Key concepts**
- **uv** is a fast Python package manager. `uv sync` reads `pyproject.toml`, creates a `.venv`
  folder (an isolated set of packages for this project, like `node_modules`) and installs
  everything. `uv run <cmd>` runs a command inside that environment. `pyproject.toml` is the
  Python equivalent of `package.json`; `uv.lock` is the equivalent of `package-lock.json`.
- **Config in one file:** every tunable number lives in `config/settings.yaml`. The loader
  (`src/config.py`) uses **pydantic**, which validates types (like Zod in TypeScript).
- **Fail safe:** values only you can supply (account size, loss limits, news window, lot value,
  spread) are `null`. `Settings.require_complete()` raises until they are filled in, so the system
  can never silently run with a guessed prop-firm rule.
- **Secrets:** API tokens go in `.env`, which is gitignored. `.env.example` shows the shape.

## Phase 1: Data

**What was built:** a downloader for 7+ years of US100 1-minute history, a cleaning step, a
resampler that builds 5m/15m/1h/4h/daily bars, a data quality report, a live price connection
(OANDA), a live economic calendar (ForexFactory) and a historical news dataset for backtests.

**Key concepts**
- **OHLC bars.** A 1-minute bar is the open/high/low/close of every price in that minute. We
  use the **bid** price, which matches what Dukascopy and OANDA publish most consistently.
- **Parquet** is a file format for tables (like a compressed CSV that remembers column types).
  It loads in a fraction of a second where CSV would take minutes. Raw data is stored one file
  per month (`data/raw/.../2024/2024-03.parquet`) with a small `manifest.json` that records
  which days downloaded. That is how `data download` knows what to skip when you re-run it.
- **UTC storage, New York logic.** Every timestamp is stored in UTC (no daylight saving).
  Session rules use New York *wall-clock* time, so 09:30 NY is 14:30 UTC in winter and
  13:30 UTC in summer automatically. The tests check both DST switches (March and November).
  The report has a "DST check" table so you can see it on real data.
- **Trading day = 18:00 to 17:00 NY.** US index CFDs follow the futures clock: the day opens
  Sunday/weekday 18:00 NY and closes 17:00 NY. Daily bars and "previous day high/low" use this
  definition. 4h bars start at 18:00, 22:00, 02:00, 06:00, 10:00 and 14:00.
- **Bar `end` time = no lookahead for higher timeframes.** A 1h bar labelled 10:00 is not
  finished until 11:00. Every resampled bar has an `end` column, and code may only use a bar
  once `end` has passed. Using the 10:00 hourly close at 10:15 would mean peeking at the future
  (**lookahead bias**). That makes backtests look far better than reality.
- **Cleaning without inventing data.** We drop duplicates, impossible bars, bars outside
  market hours and Dukascopy "filler" bars (flat, zero volume). Bad-tick wicks (a spike far
  bigger than normal that immediately reverses) are clipped. We never fill gaps: a made-up
  price could create a fake setup. Gaps are listed in the report instead.
- **Price divisor sanity check.** Dukascopy stores prices as integers. If the divisor in
  settings were wrong, every price would be 1000x off. The build refuses to continue if the
  median price is outside a plausible range.
- **REST vs streaming (OANDA).** REST is "ask for candles, get an answer". Streaming keeps one
  connection open and the server pushes every price change plus a heartbeat every few seconds.
  The heartbeat is how we can tell "quiet market" from "connection dead".
- **Mocked tests.** The cloud machine I build on can't reach Dukascopy/OANDA/ForexFactory, so
  the tests use `httpx.MockTransport`: a fake server that returns prepared responses (like
  mocking `fetch` in Jest). Your Mac run of the commands in the README is the real check.

**News data: which sources and why**
- *Live:* ForexFactory's weekly JSON feed, filtered to USD + High impact, refreshed every 30
  minutes. Every successful download is archived to `data/news/ff_archive/`, so over time we
  build our own exact record of what the live system saw.
- *Historical:* there is no free, complete, official archive of the ForexFactory calendar, so
  the history is assembled from official sources:
  1. **FRED/ALFRED (St. Louis Fed)** release dates for CPI, NFP, PPI, GDP, Retail Sales, Jobless
     Claims, PCE and JOLTS. These are the *actual* publication dates, including shifted ones
     (e.g. government shutdowns). Needs a free API key.
  2. **Federal Reserve FOMC calendar pages** for rate decisions (14:00) and press conferences (14:30).
  3. **ISM PMIs** by ISM's published rule (1st/3rd business day, 10:00), marked as rule-based.
  4. A **manual CSV** (`data/news/manual_events.csv`, columns `ts_ny,title`) for anything else.
     Fed Chair speeches are the known gap; add the important ones there.
  Treating all of these as "high impact" is deliberately conservative: more blackout, not less.
- **Fail-safe:** if the live feed fails or its data is older than `stale_after_minutes`
  (120), the news status becomes "unavailable" and the risk engine will block all signals.

## Phase 2: ICT detectors

**What was built:** every concept from the spec as a deterministic rule (`src/features/`),
the two v1 setups (`src/setups/detect.py`), 23 detector tests, and a chart tool that
draws everything on sample days for you to check.

**The one rule behind everything: `known_at`.** Every detection records the earliest moment it
could have been known live. Example: a swing high with N=2 needs two lower bars *after* it, so
it is only known when the second of those bars closes. Code may only use a detection after its
`known_at`. The backtest and the live engine call the same `detect_all()` function, so they can't
drift apart.

**How lookahead is tested.** The strongest test is simple: run the detectors on the full history,
then again on history that *stops* at time T. Everything known by T must be identical in both
runs. If any result changes when you add future data, the code was peeking. A second test cuts
the data exactly at each setup's arming time (the setup must already exist) and one bar earlier
(it must not exist yet). These tests caught two real bugs while I was building:
1. Merging two sweeps that happen close together. A later sweep was changing an earlier event,
   so merging now happens only at the MSS candle, when both are already known.
2. Choosing an FVG that only finishes forming on the bar *after* the MSS. A live run would have
   chosen differently from the backtest, so the rule now decides with what exists at the MSS
   close and only waits one bar if there is no FVG yet.

**Exact rules (all numbers live in `settings.yaml` under `detectors:` and `setups:`)**
| Concept | Rule (defaults) |
|---|---|
| ATR | Wilder ATR(14) on the setup timeframe (5m) |
| Swing | fractal pivot, N bars each side (5m N=2, 15m/1h N=3, 4h/1d N=2); confirmed N bars later |
| Liquidity | PDH/PDL (prev 18:00-17:00 day), Asia 20:00-00:00, London 02:00-05:00, overnight 18:00-09:30 high/low (known when the window ends), equal highs/lows = two swings within 0.10xATR with nothing beyond in between (valid 24h) |
| Sweep | trades >= 0.05xATR beyond the level (or closes beyond it), then closes back inside within 3 bars; one sweep per level |
| MSS | after a sweep inside a session: within 12 bars, a candle *closes* beyond the latest confirmed opposing swing (formed before the sweep) with body >= 1.0xATR. Stops if price closes back beyond the swept level |
| FVG | 3-candle gap >= 0.10xATR |
| Order block | last opposite-coloured candle at/before the sweep extreme (max 10 bars back); zone open-to-high (bear) / low-to-open (bull) |
| Premium/discount | position of the entry inside the latest confirmed 15m swing high/low range; shorts > 0.5, longs < 0.5 |
| HTF bias | daily BOS direction + 4h BOS direction + (price vs previous-day midpoint), each +1/-1; bullish if total >= 2, bearish if <= -2, else neutral |
| Setup 1 | sweep -> MSS -> limit at FVG 50% (CE) |
| Setup 2 | sweep -> MSS -> limit at OB edge, only with matching HTF bias |
| Both | stop = sweep extreme +/- 0.10xATR; target = nearest untouched opposing liquidity pool; reject if R:R < 2.0; expire after 6 bars (30 min) or at session end; premium/discount filter on both |

Every candidate setup is kept with either `valid` or the reason it was rejected
(`wrong_premium_discount`, `htf_bias_not_aligned`, `rr_below_min`, `no_target`, ...). The charts
show the rejected ones as dotted lines so you can judge the filters too.

**How to read the sample charts:** x-axis is New York time (hover a candle for Manila time).
Orange dashed = buy-side liquidity (highs), blue dashed = sell-side (lows), dotted = equal
highs/lows; a line ends where the level was taken. Orange X = sweep, white line = the swing
the MSS closed through, purple box = order block, green/red boxes = bullish/bearish FVGs,
blue/red/green lines = entry/stop/target of a setup. The setup list is under the chart.
Click legend items to hide layers.

## First run on real data (2026-10-16)

**What changed:** until now the code had only run on synthetic data. This run used the real
Dukascopy US100 history (2019-01 to 2026-10, 2.66M clean 1-minute bars).

- **Real data matched the assumptions.** The binary candle format, the price divisor (1000)
  and the filler-bar rule were all right. DST is handled: the NY 09:30 open is 14:30 UTC in
  winter and 13:30 UTC in summer.
- **Downloads are flaky, so they resume.** Dukascopy times out on some days. A month with any failed
  day is marked incomplete and retried on the next run, so you re-run the command until
  nothing is incomplete.
- **Fail-safe news feed:** a rate-limited, non-JSON or *empty* feed raises an error instead of
  meaning "no news". An empty list looks like "safe to trade", which is the dangerous reading.
- **Why 145 setups is not a bug.** Over 7.8 years there were 16,692 sweeps, 684 MSS and 145
  valid setups (about 1.5 per month). Each filter only removes candidates. That is the honest output of the rules as
  specified. Do not loosen filters to get more trades; that is how you overfit.
- **Setups armed at the very end of a session** (16 of 145 have 5 minutes or less to fill) can
  barely fill. This is a rule question for Rald, not something to patch silently.

## Phase 3: Backtester and prop rules

**What was built**
- `src/risk/sizing.py`: lots = (initial balance × 0.5%) / (stop points × $ per point), rounded
  **down** to 0.01. If even 0.01 lots risks too much, the signal is blocked. It is never rounded up.
- `src/risk/guard.py`: the hard gate every signal passes, in the backtest now and live later. It
  lists *every* reason a signal is blocked: news, session, daily buffer, drawdown halt, signal caps,
  2 losses in a row, an open position, sizing, "a full loss could breach a firm limit", and later
  model confidence.
- `src/backtest/fills.py`: replays each order on 1-minute bars.
- `src/backtest/engine.py`: a **raw** run (every valid setup, costs only) and a **guarded** run
  (one account, all rules, exactly as live).
- `src/backtest/montecarlo.py`: the chance of passing the 2-Step challenge.
- `uv run python -m src.cli backtest run` writes `reports/backtest_latest.html`, plus trades, blocks
  and events CSVs in `reports/backtest/`.

**Key concepts**
- **R (risk multiple):** profit divided by what you planned to risk. +2R means you made twice your
  risk; -1R means a normal stop-out. Working in R makes results independent of account size and
  lot value.
- **Expectancy:** the average R per trade. If it is above 0 after costs, the setup makes money on
  average. One good year doesn't prove it. Look at the **t-stat** too: below about 2, the result can't be
  told apart from luck.
- **Bid/ask:** the data is the *bid* price. Buying costs the *ask* (bid + spread), so a buy-limit
  only fills when the ask reaches it, and a short is stopped when the ask hits the stop.
- **Pessimistic fills:** one minute bar only gives open/high/low/close, not the order inside it.
  When the stop and target are both inside one bar, we assume the stop came first. The backtest
  should be harder than reality, not easier.
- **News rule:** Equity Edge forbids opening *or closing* trades inside the news window. The
  backtest therefore closes positions and cancels orders 1 minute before a window starts.
- **Monte Carlo:** we replay random sequences of real historical days thousands of times and
  count how often the account reaches +8% (then +5%) before hitting a limit. It is only as good as
  the trades it samples.
- **The gate:** ML can filter a real edge, but it can't create one. If the raw setups lose
  money, a model trained on them mostly learns noise. That is why we stop here and decide together.

## Phase 3b: Rule research with a locked holdout

**Why a holdout:** if you try many rule variants on the same data and keep the best one, you
have mostly found the luckiest one. So 2025-01-01 onward is **locked**. All research uses
2019-2024 only, and the holdout is used **once**, on the single variant you choose. If the variant
fails there, we don't go back and tweak it. That would turn the holdout into more training data.

**Pre-registration:** the 9 variants in `settings.yaml -> research.variants` were written before
any of them was run, and every one is reported, including the bad ones.

**Your rule:** `setups.min_rr` must be >= 1.0. The config refuses to load otherwise, so no
setting can produce a trade targeting less than 1R.

**Multiple testing:** with 9 variants, one will probably look "good" by chance. A t-stat around
1 is what luck alone looks like. We want about 2.5 or more before trusting a result.

Run it with: `uv run python -m src.cli backtest research` (report: `reports/research_latest.html`).

**Holdout result (2026-10-16):** you picked `fixed_1_5r_target`. On the untouched 2025-2026 data
it made **-0.11 R per trade** over 17 trades (win rate 35%, t-stat -0.40). In-sample it showed
+0.13 R. That is the classic shape of a result that was luck: decent in-sample, gone out of
sample. Every variant shown so far fails that test, so the honest conclusion is that **these ICT rules,
as defined, show no reliable edge on US100**. The holdout is now spent. A new idea must prove
itself on new data (paper trading) before real money.
