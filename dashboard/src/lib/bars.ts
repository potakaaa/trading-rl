import type { Bar, Timeframe } from "./api";
import { isoToSec } from "./format";

export const TF_SECONDS: Record<Timeframe, number> = { "1m": 60, "5m": 300, "15m": 900 };

/** Floor a unix-seconds timestamp to the start of its timeframe bucket. */
export function bucketStart(sec: number, tf: Timeframe): number {
  const size = TF_SECONDS[tf];
  return Math.floor(sec / size) * size;
}

/**
 * Merge a completed 1m bar into a bar list of timeframe `tf` (ascending).
 * Updates the forming higher-TF candle or appends a new one. Pure.
 */
export function mergeMinuteBar(bars: Bar[], minute: Bar, tf: Timeframe): Bar[] {
  const bucket = bucketStart(isoToSec(minute.t), tf);
  const bucketIso = new Date(bucket * 1000).toISOString().replace(".000Z", "Z");
  const last = bars[bars.length - 1];
  if (!last) return [{ ...minute, t: bucketIso }];
  const lastSec = isoToSec(last.t);
  if (bucket < lastSec) return bars; // stale/out-of-order message
  if (bucket === lastSec) {
    if (tf === "1m") return [...bars.slice(0, -1), { ...minute, t: last.t }];
    const merged: Bar = {
      t: last.t,
      o: last.o,
      h: Math.max(last.h, minute.h),
      l: Math.min(last.l, minute.l),
      c: minute.c,
    };
    return [...bars.slice(0, -1), merged];
  }
  return [...bars, { ...minute, t: bucketIso }];
}
