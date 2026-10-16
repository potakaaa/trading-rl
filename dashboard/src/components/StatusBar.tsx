"use client";

import type { StatusResponse } from "../lib/api";
import { TZ_MANILA, TZ_NY, fmtNum, fmtTime } from "../lib/format";
import { StatusBadge } from "./ui";

export function PaperBanner() {
  return (
    <div className="rounded-lg border border-amber-500/50 bg-amber-500/15 px-3 py-2 text-center text-sm font-semibold text-amber-200">
      PAPER MODE – signals are unproven (backtest showed no edge)
    </div>
  );
}

function lagTone(lag: number | null, ok: boolean): string {
  if (!ok || lag === null) return "text-red-300";
  if (lag > 120) return "text-red-300";
  if (lag > 60) return "text-amber-300";
  return "text-emerald-300";
}

export function StatusBar({
  status,
  wsConnected,
  now,
}: {
  status: StatusResponse;
  wsConnected: boolean;
  now: number | null;
}) {
  const e = status.engine;
  return (
    <div className="space-y-2">
      {status.mode === "paper" ? <PaperBanner /> : null}
      <div className="flex flex-col gap-3 rounded-lg border border-edge bg-panel p-3 md:flex-row md:items-center md:justify-between">
        <div className="flex min-w-0 flex-wrap items-center gap-3">
          <StatusBadge status={status.status} />
          {status.status_reason ? (
            <span className="min-w-0 text-sm text-slate-300">{status.status_reason}</span>
          ) : null}
        </div>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-3 lg:flex lg:flex-wrap lg:gap-x-6">
          <Item label="Mode" value={status.mode.toUpperCase()} />
          <Item label="Session" value={status.session ?? "none"} />
          <Item label="Price" value={status.price === null ? "–" : fmtNum(status.price)} />
          <Item
            label="Feed lag"
            value={
              <span className={lagTone(e.feed_lag_seconds, e.feed_ok)}>
                {e.feed_lag_seconds === null ? "–" : `${e.feed_lag_seconds.toFixed(0)}s`}
                {e.feed_ok ? "" : " (down)"}
              </span>
            }
          />
          <Item
            label="Engine"
            value={<span className={e.running ? "text-emerald-300" : "text-red-300"}>{e.running ? "running" : "stopped"}</span>}
          />
          <Item
            label="Live feed"
            value={<span className={wsConnected ? "text-emerald-300" : "text-amber-300"}>{wsConnected ? "connected" : "reconnecting"}</span>}
          />
          <Item
            label="Now"
            value={now === null ? "–" : `NY ${fmtTime(now, TZ_NY)} · MNL ${fmtTime(now, TZ_MANILA)}`}
          />
        </dl>
      </div>
      {e.feed_error ? (
        <p className="rounded border border-red-500/30 bg-red-500/10 px-3 py-1.5 text-xs text-red-300">Feed error: {e.feed_error}</p>
      ) : null}
      {status.warnings.length ? (
        <ul className="space-y-1 rounded border border-amber-500/25 bg-amber-500/5 px-3 py-2 text-xs text-amber-200/90">
          {status.warnings.map((w, i) => (
            <li key={i}>⚠ {w}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function Item({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[10px] uppercase tracking-wider text-muted">{label}</dt>
      <dd className="truncate font-medium tabular-nums">{value}</dd>
    </div>
  );
}
