# Round 5: published setups of documented traders/researchers. Pre-registered 2026-10-17, before any test

Rald asked for setups used by successful traders, applied to our system. Private journals are not
public, so this round uses **published, codeable methods** with documented records. They are
implemented as published (US100 replaces SPY/QQQ, since the CFD tracks the Nasdaq-100), then judged
by our costs, our prop rules and Rald's $5 rule.

## Common rules
- US100 Dukascopy BID 1m (same feed as live). Research 2019-01-01 .. 2026-04-15. **Final test
  2026-04-16 .. 2026-10-15** (still untouched), run once, only for a setup that passes.
- Costs: spread 2.0 pts, stop slippage 1.0 (2x stress = 4/2). Entries at market (next 1m open after the
  signal) unless the rule uses a stop order. Stop orders fill at the trigger price plus slippage in the
  trade's direction, or at the bar open if price gaps through. Flat by 16:00 NY (these published
  setups hold through the lunch hour; nothing is held overnight).
- Risk 0.5% ($12.50) per trade, R = net P&L / initial stop distance. Equity Edge guard and news
  blackout apply in the guarded run (signals inside a blackout are skipped).
- Every setup's parameters are the published defaults below. Nothing is tuned.

## Setups
1. `noise_momentum` (Zarattini, Aziz & Barbon 2024, "Beat the Market"):
   - **Noise boundaries:** for each minute-of-day m, sigma_m = mean over the previous 14 days of |close_m / open_0930 - 1|.
     Upper boundary = max(today's 09:30 open, previous 16:00 close) x (1 + sigma_m). Lower boundary =
     min(open, previous close) x (1 - sigma_m).
   - **Checks:** at every :00 and :30 from 10:00 to 15:30. Price above the upper boundary -> long; below the
     lower boundary -> short; reverse if the opposite side is crossed.
   - **Exit:** trailing stop = max(upper boundary, VWAP) for longs (min(lower boundary, VWAP) for shorts),
     checked at the same times. Flat at 16:00.
   - **Initial risk** = distance from entry to the trailing stop at entry.
2. `orb5` (Zarattini & Aziz 2023, "Can Day Trading Really Be Profitable?"):
   - **Entry:** if the 09:30-09:35 candle closes up, go long at 09:35; if it closes down, go short. A doji means no trade.
   - **Stop:** the low (long) or high (short) of that candle. **Target:** 10R. Otherwise exit at 16:00.
3. `crabel_nr7` (Toby Crabel, *Day Trading with Short Term Price Patterns and ORB*):
   - **When:** only on days after an NR7 day (yesterday's range is the narrowest of the last 7 days).
   - **Stretch** = mean over the last 10 days of min(open - low, high - open).
   - **Entry:** buy stop at open + stretch, sell stop at open - stretch, whichever triggers first after
     09:30. The stop is at the other trigger level. Exit at 16:00. One trade per day.
4. `holy_grail` (Linda Raschke, *Street Smarts*), on 5m bars:
   - **When:** ADX(14) > 30 and rising, and a bar touches the 20 EMA.
   - **Entry:** buy stop above that bar's high (in an uptrend, EMA rising) or sell stop below its low
     (downtrend), valid for the next 3 bars.
   - **Stop:** the pullback extreme. **Target:** the high (low) of the last 20 bars before the pullback,
     skipped if under 1R.
   - **Hours:** 10:00-15:30 entries. Max 2 trades per day.
5. `williams_vbo` (Larry Williams volatility breakout):
   - **Entry:** buy stop at open + 0.5 x yesterday's range, sell stop at open - 0.5 x yesterday's range
     (09:30 open, previous 18:00-17:00 day range).
   - **Stop:** the 09:30 open. Exit at 16:00. One trade per day.

## Pass criteria (raw run on the research period, all must hold)
1. at least 150 trades
2. **average net profit per trade > $5, i.e. mean R > 0.4 at base costs (Rald's rule)**, and mean R > 0 with 2x costs
3. t-stat of mean R >= 3.0
4. positive total R in at least 5 of the 7 full years 2019-2025

Reported for every setup: win rate (> $5 = win), average win and loss in $, profit factor, results by year,
and the published claim for comparison. A setup that passes runs once on the final six months. If
it holds up there too, it goes to paper trading.
