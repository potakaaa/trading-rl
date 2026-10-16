"use client";

import type { AlertLevel, BlockEntry } from "../lib/api";
import { TZ_MANILA, TZ_NY, fmtTime } from "../lib/format";
import { Panel, Pill } from "./ui";

export function BlocksPanel({ blocks, error }: { blocks: BlockEntry[] | undefined; error?: string }) {
  return (
    <Panel title={`Today's blocks${blocks ? ` · ${blocks.length}` : ""}`}>
      {error ? <p className="mb-2 text-xs text-red-300">{error}</p> : null}
      {!blocks ? (
        <p className="text-sm text-muted">Loading…</p>
      ) : blocks.length === 0 ? (
        <p className="text-sm text-muted">No setups blocked today.</p>
      ) : (
        <ul className="max-h-72 space-y-1.5 overflow-y-auto pr-1 text-xs">
          {blocks.map((b, i) => (
            <li key={`${b.time}-${i}`} className="rounded border border-edge/70 px-2 py-1.5">
              <div className="flex flex-wrap items-center gap-2">
                <span className="tabular-nums text-muted">
                  NY {fmtTime(b.time, TZ_NY)} · MNL {fmtTime(b.time, TZ_MANILA)}
                </span>
                <span className="font-medium">{b.setup_type}</span>
                <span className={b.direction === "long" ? "text-emerald-300" : "text-red-300"}>{b.direction}</span>
              </div>
              <div className="mt-1 flex flex-wrap gap-1">
                {b.reasons.map((r) => (
                  <Pill key={r} tone="amber">
                    {r}
                  </Pill>
                ))}
              </div>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export interface AlertItem {
  id: number;
  level: AlertLevel;
  text: string;
  at: number;
}

const LEVEL_STYLE: Record<AlertLevel, string> = {
  info: "border-sky-500/40 bg-sky-950/90 text-sky-100",
  warning: "border-amber-500/50 bg-amber-950/90 text-amber-100",
  critical: "border-red-500/60 bg-red-950/95 text-red-100",
};

export function AlertFeed({ alerts }: { alerts: AlertItem[] }) {
  return (
    <Panel title="Alerts">
      {alerts.length === 0 ? (
        <p className="text-sm text-muted">No alerts since this page opened.</p>
      ) : (
        <ul className="max-h-60 space-y-1 overflow-y-auto pr-1 text-xs">
          {alerts.map((a) => (
            <li key={a.id} className="flex gap-2">
              <span className="shrink-0 tabular-nums text-muted">{fmtTime(a.at, TZ_NY)}</span>
              <span
                className={
                  a.level === "critical" ? "text-red-300" : a.level === "warning" ? "text-amber-300" : "text-slate-300"
                }
              >
                {a.text}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export function Toasts({ toasts, onDismiss }: { toasts: AlertItem[]; onDismiss: (id: number) => void }) {
  if (toasts.length === 0) return null;
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-50 flex flex-col items-center gap-2 p-3 sm:items-end">
      {toasts.map((t) => (
        <div
          key={t.id}
          role="status"
          className={`pointer-events-auto flex w-full max-w-sm items-start gap-2 rounded-md border px-3 py-2 text-sm shadow-lg ${LEVEL_STYLE[t.level]}`}
        >
          <span className="text-[10px] font-bold uppercase tracking-wider opacity-80">{t.level}</span>
          <span className="flex-1">{t.text}</span>
          <button type="button" className="text-xs opacity-70 hover:opacity-100" onClick={() => onDismiss(t.id)} aria-label="Dismiss">
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
