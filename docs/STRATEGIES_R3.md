# Strategy research round 3 (swing): pre-registration (written 2026-10-17, BEFORE any test)

Rald relaxed the constraints for this round: multi-day holds including weekends (a third-party
source says Equity Edge 2-Step allows this; **confirm with Equity Edge before trading**), and three
more instruments. This file is committed before any round-3 strategy is backtested and before the new
instruments' data has been analysed. Rules, parameters and pass criteria are frozen.

## Instruments and costs (UNCONFIRMED conservative estimates, settings.yaml -> instruments)
| | spread | stop slippage | swap per calendar night (long and short both pay) |
|---|---|---|---|
| US100 | 2.0 pts | 1.0 | 7.5%/360 of price |
| US500 | 0.6 pts | 0.25 | 7.5%/360 of price |
| XAUUSD | 0.30 $ | 0.10 | 7.5%/360 of price |
| EURUSD | 0.00010 | 0.00005 | 0.0001 (1 pip) |
A 2x stress run doubles spread, slippage AND swap.

## Common rules
- Data: Dukascopy BID 1m (same feed as the live engine), daily bars = trading day 18:00-17:00 NY.
  Research 2019-01-01 .. 2024-12-31 (2018 only as warm-up for the averages). 2025-01-01 onward is
  untouched until the single final test.
- Signals use the **daily close** (bar complete at 17:00 NY). Every entry and every rule-based exit
  is a **market order at 09:30 NY on the next trading day** (21:30 Manila), long pays the spread.
  Stops (and targets where defined) are live orders, simulated on 1m bars with the same
  pessimistic rules as before (gaps through a stop fill at the gap price).
- Risk 0.5% of the initial balance per trade, sized from the stop distance. One position per
  strategy per instrument at a time. R is measured from the actual fill. ATR = Wilder ATR(20) on daily bars.
- Rald's 1R rule: a **fixed** target, where one exists, is never below 1R. Trend and momentum systems
  have no fixed target. They exit by their rule or stop, so the possible reward is open-ended.

## Strategies (all instruments unless stated)
| id | entry (at a daily close) | stop | exit |
|---|---|---|---|
| `donchian` | long: close > highest high of the previous 20 days; short: close < lowest low of the previous 20 days | 2 x ATR from entry | close crosses the 10-day channel the other way (long: close < 10-day low), or stop |
| `ma_trend` | SMA(50) crosses above SMA(200) -> long; crosses below -> short | 3 x ATR | the opposite cross, or stop |
| `rsi2_revert` | long: close > SMA(200) and RSI(2) < 10; short: close < SMA(200) and RSI(2) > 90 | 2 x ATR | target 1R, or after 10 trading days, or stop |
| `tsmom` | every Friday close: long if the 63-day return > 0, short if < 0 | 3 x ATR | re-decided each Friday (exit/reverse when the sign flips), or stop |
| `turn_of_month` (US100, US500 only) | long at the close of the second-to-last trading day of the month (entry next 09:30, i.e. the last trading day) | 2 x ATR | after the 3rd trading day of the new month (exit 09:30 on the 4th), or stop |
| `overnight` (US100, US500 only) | long at the 15:55 NY close every trading day (market order at 15:55 close, not 09:30) | 1 x ATR | 09:35 NY next trading day, or stop |

## Pass criteria (raw, 2019-2024, per strategy pooled over its instruments, all must hold)
1. >= 60 filled trades
2. expectancy > 0 R with the 2x cost stress (spread, slippage, swap)
3. t-stat of mean R >= 2.0 at base costs
4. positive total R in >= 4 of the 6 years
Reported for information: win rate, profit factor, per-instrument results, the share of exits that
would fall inside a news blackout window, and a portfolio simulation (all passing strategies
together, max 4 open positions, Equity Edge daily/max-drawdown checks with daily marked equity).

## RL trade filter
The round-2 RL filter (fitted Q-iteration, same reward, actions skip/half/full) is re-run on the pooled
round-3 candidates with daily features (strategy, instrument, direction, ATR percentile, 20/63-day
return in ATR, distance to SMA200 in ATR, stop size in ATR, weekday). Episodes = calendar weeks.
Walk-forward expanding window, 6-month test folds 2021H1..2024H2. Same RL pass rule as round 2.

## Final test (once)
Passing strategies (and the RL filter if it passes) are run once on 2025-01-01 .. latest. If nothing
passes, nothing is run on that period.
