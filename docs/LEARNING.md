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

## Phase 5: Live engine and alerts (paper only)

**What was built**
- `src/data/dukascopy_live.py`: live 1-minute bars from Dukascopy's public chart endpoint. It is
  the **same feed as our history**: a check matched 120 stored bars exactly (0.0 difference,
  including volume). New bars arrive about 10-20 s after each minute closes.
- `src/live/engine.py`: every 15 s it fetches new bars, tracks open paper signals, and on every
  completed 5-minute bar inside a session runs **the same** `detect_all` and `guard.evaluate` as
  the backtest.
- `src/live/store.py`: SQLite at `data/live/live.db` holding signals, every block with its reasons
  (your compliance record), alerts and settings.
- `src/alerts/notify.py`: Telegram and Discord. A missing key is skipped, and a failing alert
  never crashes the engine.
- `src/api/app.py`: FastAPI REST + WebSocket (contract in `docs/API.md`).
- One command starts it all: `uv run python -m src.cli live run`.

**Key concepts**
- **Parity:** "backtest and live use the same code" is tested, not assumed. `tests/test_live.py`
  replays days minute by minute through the live engine and checks that it proposes exactly the
  setups the backtest found, at the same time and price. Three real 2023 days also matched.
- **Fail safe:** if the feed errors or goes quiet for more than 90 s in a session, the status is NO-DATA and
  there are no signals, plus an alert. If news can't be loaded or is older than 2 h: NO-NEWS, no signals.
- **Late data:** if the feed drops out and a 5m bar arrives more than 2 minutes late, its setup
  is blocked (`late_data`). The entry may already be gone, so a late signal is worse than none.
  The live test on 2026-10-16 had two 1.5-2 minute outages right at the 09:30 open. They triggered
  NO-DATA and alerts as designed.
- **Paper mode:** every signal is labelled unproven. The engine simulates each signal's fill and
  exit with the backtest's fill rules, and those paper results drive the account state by default
  (`live.state_source: paper`). Switch to `journal` to use only the trades you mark as taken.
- **Restarts are safe:** account state is rebuilt from the database on every decision, so a
  restart neither forgets today's losses nor invents trades.

**Keeping the Mac awake:** `live run` starts `caffeinate -i`, which stops idle sleep while the
engine runs. **Closing the lid still sleeps a MacBook.** Keep it plugged in and open, and in
System Settings > Battery > Options turn on "Prevent automatic sleeping when the display is off".
Check with `pmset -g` (look for `sleep 0`).

### Telegram bot setup
1. In Telegram, open a chat with **@BotFather** and send `/newbot`. Pick a name and a username
   ending in `bot`.
2. BotFather replies with a token like `123456:ABC...`. Put it in `.env` as `TELEGRAM_BOT_TOKEN=`.
3. Send any message to your new bot, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy `"chat":{"id":...}`
   into `.env` as `TELEGRAM_CHAT_ID=`.
4. Restart `live run`. The startup checklist shows `[OK] Telegram alerts`.

### Discord webhook setup
1. In your Discord server: channel settings (gear) > Integrations > Webhooks > New Webhook.
2. Copy the webhook URL into `.env` as `DISCORD_WEBHOOK_URL=`, then restart `live run`.

Never commit `.env` (it is gitignored). Anyone with the token or webhook URL can post as your bot.

## Phase 6: Dashboard

`cd dashboard && npm install && npm run dev`, then open http://localhost:3000 with the backend
running. It shows the chart (1m/5m/15m with liquidity, FVG, OB, sweep and MSS overlays), the
signal panel (Taken / Skipped plus a fill and exit form), the risk panel, news in New York and Manila
time, today's blocks, plus Backtest, Journal and Settings pages.

## Phase 7: Paper trading

Run `live run` during NY sessions for **at least 4 weeks**. Every Friday after the close the engine
writes `reports/paper_latest.html` and sends a summary alert. You can also run
`uv run python -m src.cli live paper-report` at any time.

**How drift is judged:** if paper trades behave like the backtest, the average R of n trades should
land within ±2 × (backtest spread of R) / √n of the backtest average about 95% of the time. Outside
that band = DRIFT, meaning live differs from what we tested. With about 0.5 signals per week, 4 weeks
gives only a few trades. That is too few to prove anything either way, which is why the verdict says so.

## Round 2: seven more strategies + a reinforcement-learning trade filter (2026-10-16)

**What was tested:** opening-range breakout (15 and 30 min), VWAP band fade, opening momentum,
"first half hour predicts last half hour" momentum, overnight gap fade and prior-day high/low
breakout. All entered at market with the same costs, sessions and prop rules as before. The rules
and pass criteria were committed **before** any test ran (`docs/STRATEGIES.md`, commit `de60517`),
so they could not be adjusted to fit the results.

**Result: none passed.** Every strategy lost money on 2019-2024 (3,300+ trades for the bigger
ones), including before costs. VWAP fade -0.30 R per trade (t -8.8), first/last half-hour momentum
-0.13 R (t -5.5), gap fade -0.39 R. The best was prior-day breakout at -0.06 R (+0.06 R before
costs, too small to survive a 2-point spread).

**Why costs matter so much:** a strategy with a 15-point stop pays 2 points of spread per trade.
That is about 0.13 R gone before the market moves. Short-stop intraday ideas need a very real edge
to beat that.

**The RL filter** (fitted Q-iteration, `src/ml/rl_filter.py`) learns, walk-forward, whether to skip,
take at half size or take at full size for each candidate. It is rewarded with the R it makes,
penalised more for losses, and penalised again for drawing the day down by 2R or more.
- **Synthetic checks:** to show the RL works, `tests/test_rl_filter.py` plants an edge in fake data.
  The agent finds it and profits out of sample. On pure noise it learns to skip.
- **Real data:** it learned to skip almost everything (91 trades out of about 3,000 candidates in 4 years).
  It beat "take everything" in all 8 test periods, but only by losing less: -5.1 R total out of sample.
- **Conclusion:** there is no profitable pocket inside these strategies for it to find. Skipping was
  the honest best answer.

**Win rate vs profit:** the highest win rate was momentum at 48%, and it still lost (wins of 1R vs
losses of slightly more after costs). Accuracy alone never makes a strategy profitable.

Per the pre-registration, the final 2025-2026 test was **not** run, because nothing qualified for it.

Run everything again with `uv run python -m src.cli backtest round2` (report:
`reports/round2_latest.html`). LightGBM on macOS needs `brew install libomp` (already installed).

## Round 3: swing trading on 4 instruments + RL filter (2026-10-17)

**What changed:** holds can now last overnight and over weekends, and US500, gold (XAUUSD) and EURUSD were added.
Rules were frozen first in `docs/STRATEGIES_R3.md` (commit `898fe26`). Data: 2018-2026 1m bars for
all four instruments from Dukascopy's chart endpoint (identical, bar for bar, to the original US100
history). Costs include an UNCONFIRMED swap estimate per night held.

**Strategies (2019-2024): none passed.**
- **Closest:** turn-of-month (long US indices around month-end) made +0.095 R per trade over 142 trades,
  with a 60% win rate, and stayed positive with doubled costs. Its t-stat of 1.6 is below the 2.0
  bar, so it can't be told apart from luck yet.
- **Swap costs:** they killed the long-hold trend systems. Moving-average trend made +0.85 R per trade
  before costs but lost after them, because it holds for weeks and pays swap every night.

**RL filter: CORRECTION (2026-10-17).** The first run reported that the RL filter passed and then made
+12 R in a one-time 2025-26 test. **That result was wrong.** The agent's "R so far" state included
the final result of trades that were *still open* when it made its next decision. With multi-day
swing trades, that is lookahead: it was partly seeing the future. After the fix, a trade only counts
once it has closed, and a regression test now checks this.
- **Corrected walk-forward:** -2.1 R total out of sample (-0.06 R per unit with doubled costs). It
  beat "take every trade" in only 4 of 8 periods, so it **fails** the pre-registered rule.
- **No final test:** because it fails, the corrected filter was not tested on 2025-26. The earlier
  final-test file is kept as `reports/round3/final_test_INVALID_lookahead.json` for the record.
- **Round 2:** its RL filter was re-run with the fix and still fails (-13.2 R).

**Lesson:** a result that looks good on unseen data can still come from a bug. Lookahead is easy to
introduce, which is why every state and feature needs a "could I have known this then?" test.

## Round 4: intraday machine learning (2026-10-17)

**Setup** (pre-registered in `docs/STRATEGIES_R4.md`): 10 decision moments per NY session day, a
fixed trade (stop 1 x ATR of 15-minute bars, target 1.5R, out after 60 min), 22 features known at
the decision time, and labels simulated with the same fill code as every backtest.
- **Stage 1, volatility gate:** a LightGBM model predicts how active the next hour will be.
- **Stage 2, direction:** LightGBM models predict the net R of going long and of going short.
- **Validation:** 11 walk-forward folds from 2021 to April 2026.

**Result: fail, but informative.**
- **The model has real skill:** its trades averaged +0.005 R, versus -0.107 R for the same moments with a
  random direction. That 0.11 R gap is statistically solid (t about 3).
- **Costs cancel it:** a 2-point spread plus 1 point of stop slippage on a 1 x ATR15 stop (about 30 points)
  costs roughly the same 0.11 R. Net result: about zero.
- **The volatility gate hurt:** the most active moments were not easier to predict, and costs stayed the same.
- **What follows:** per the pre-registration, there is no neural network on this setup, and the final six months
  (2026-04-16 .. 2026-10-15) are still untouched.

**The lesson:** prediction skill alone is not enough. It has to be bigger than the cost of each trade.
The two levers are wider stops (so costs are a smaller share of R) and lower costs (a tighter spread).
