export const TZ_NY = "America/New_York";
export const TZ_MANILA = "Asia/Manila";

const fmtCache = new Map<string, Intl.DateTimeFormat>();

function formatter(tz: string, opts: Intl.DateTimeFormatOptions): Intl.DateTimeFormat {
  const key = tz + JSON.stringify(opts);
  let f = fmtCache.get(key);
  if (!f) {
    f = new Intl.DateTimeFormat("en-US", { timeZone: tz, hour12: false, ...opts });
    fmtCache.set(key, f);
  }
  return f;
}

function toDate(v: string | number | Date): Date | null {
  const d = v instanceof Date ? v : new Date(v);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** "14:05" in the given zone. */
export function fmtTime(v: string | number | Date | null | undefined, tz: string): string {
  if (v === null || v === undefined) return "–";
  const d = toDate(v);
  if (!d) return "–";
  return formatter(tz, { hour: "2-digit", minute: "2-digit" }).format(d);
}

/** "Fri Oct 16, 14:05" in the given zone. */
export function fmtDateTime(v: string | number | Date | null | undefined, tz: string): string {
  if (v === null || v === undefined) return "–";
  const d = toDate(v);
  if (!d) return "–";
  return formatter(tz, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(d);
}

/** "Oct 16" in the given zone. */
export function fmtDay(v: string | number | Date, tz: string): string {
  const d = toDate(v);
  if (!d) return "–";
  return formatter(tz, { weekday: "short", month: "short", day: "numeric" }).format(d);
}

/** YYYY-MM-DD calendar date in the given zone. */
export function ymd(v: string | number | Date, tz: string): string {
  const d = toDate(v);
  if (!d) return "";
  const parts = formatter(tz, { year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(d);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  return `${get("year")}-${get("month")}-${get("day")}`;
}

/** Both zones: "NY 09:45 · MNL 21:45". */
export function fmtBoth(v: string | null | undefined): string {
  if (!v) return "–";
  return `NY ${fmtDateTime(v, TZ_NY)} · MNL ${fmtTime(v, TZ_MANILA)}`;
}

export function fmtCountdown(ms: number): string {
  const neg = ms < 0;
  let s = Math.floor(Math.abs(ms) / 1000);
  const d = Math.floor(s / 86400);
  s -= d * 86400;
  const h = Math.floor(s / 3600);
  s -= h * 3600;
  const m = Math.floor(s / 60);
  s -= m * 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  const body = d > 0 ? `${d}d ${h}h ${pad(m)}m` : h > 0 ? `${h}h ${pad(m)}m ${pad(s)}s` : `${m}m ${pad(s)}s`;
  return neg ? `-${body}` : body;
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtUsd(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const sign = v < 0 ? "-" : "";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

export function fmtSigned(v: number | null | undefined, digits = 2, suffix = ""): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}${suffix}`;
}

/** ISO string -> unix seconds. */
export function isoToSec(iso: string): number {
  return Math.floor(new Date(iso).getTime() / 1000);
}
