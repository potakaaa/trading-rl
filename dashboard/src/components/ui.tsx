"use client";

import { useState, type ReactNode } from "react";
import type { EngineStatusCode } from "../lib/api";

export function Panel({
  title,
  right,
  children,
  className = "",
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`min-w-0 rounded-lg border border-edge bg-panel ${className}`}>
      <header className="flex items-center justify-between gap-2 border-b border-edge px-3 py-2">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">{title}</h2>
        {right}
      </header>
      <div className="p-3">{children}</div>
    </section>
  );
}

const STATUS_STYLE: Record<EngineStatusCode, string> = {
  ACTIVE: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/40",
  "PAUSED-NEWS": "bg-amber-500/15 text-amber-300 ring-amber-500/40",
  "STOPPED-FOR-DAY": "bg-orange-500/15 text-orange-300 ring-orange-500/40",
  HALTED: "bg-red-500/20 text-red-300 ring-red-500/50",
  "NO-DATA": "bg-red-500/15 text-red-300 ring-red-500/40",
  "NO-NEWS": "bg-amber-500/15 text-amber-300 ring-amber-500/40",
  "OUTSIDE-SESSION": "bg-slate-500/15 text-slate-300 ring-slate-500/40",
};

export function StatusBadge({ status }: { status: EngineStatusCode | string }) {
  const style =
    STATUS_STYLE[status as EngineStatusCode] ?? "bg-slate-500/15 text-slate-300 ring-slate-500/40";
  return (
    <span
      className={`inline-flex items-center rounded-md px-3 py-1.5 text-lg font-bold tracking-wide ring-1 ${style}`}
    >
      {status}
    </span>
  );
}

export function Pill({ children, tone = "slate" }: { children: ReactNode; tone?: Tone }) {
  return (
    <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-medium ring-1 ${TONES[tone]}`}>
      {children}
    </span>
  );
}

export type Tone = "slate" | "green" | "red" | "amber" | "blue" | "violet";
const TONES: Record<Tone, string> = {
  slate: "bg-slate-500/10 text-slate-300 ring-slate-500/30",
  green: "bg-emerald-500/10 text-emerald-300 ring-emerald-500/30",
  red: "bg-red-500/10 text-red-300 ring-red-500/30",
  amber: "bg-amber-500/10 text-amber-300 ring-amber-500/30",
  blue: "bg-sky-500/10 text-sky-300 ring-sky-500/30",
  violet: "bg-violet-500/10 text-violet-300 ring-violet-500/30",
};

export function DirectionPill({ direction }: { direction: string }) {
  const long = direction === "long" || direction === "bull";
  return <Pill tone={long ? "green" : "red"}>{direction.toUpperCase()}</Pill>;
}

/**
 * Progress bar scaled to the firm limit, with a tick at the internal buffer.
 */
export function LimitBar({
  label,
  used,
  buffer,
  limit,
}: {
  label: string;
  used: number;
  buffer: number;
  limit: number;
}) {
  const safeLimit = limit > 0 ? limit : 1;
  const pct = Math.max(0, Math.min(100, (used / safeLimit) * 100));
  const bufPct = Math.max(0, Math.min(100, (buffer / safeLimit) * 100));
  const color =
    used >= buffer ? "bg-red-500" : used >= buffer * 0.6 ? "bg-amber-400" : "bg-emerald-500";
  return (
    <div>
      <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-2 text-xs">
        <span className="text-muted">{label}</span>
        <span className="tabular-nums">
          <span className="font-semibold text-fg">${used.toFixed(2)}</span>
          <span className="text-muted">
            {" "}
            / buffer ${buffer.toFixed(0)} / firm ${limit.toFixed(0)}
          </span>
        </span>
      </div>
      <div className="relative h-2.5 w-full overflow-hidden rounded bg-white/5">
        <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
        <div
          className="absolute top-0 h-full w-0.5 bg-amber-200/80"
          style={{ left: `calc(${bufPct}% - 1px)` }}
          title={`internal buffer $${buffer}`}
        />
      </div>
    </div>
  );
}

export function BackendOffline({ detail }: { detail?: string }) {
  return (
    <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm">
      <p className="font-semibold text-red-300">Backend offline</p>
      <p className="mt-1 text-red-200/90">
        Start it with{" "}
        <code className="rounded bg-black/40 px-1.5 py-0.5 font-mono text-red-100">
          uv run python -m src.cli live run
        </code>
        . This page retries automatically.
      </p>
      {detail ? <p className="mt-1 text-xs text-red-200/60">{detail}</p> : null}
    </div>
  );
}

export function ErrorNote({ text }: { text: string }) {
  return <p className="rounded border border-red-500/30 bg-red-500/10 px-2 py-1 text-xs text-red-300">{text}</p>;
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <p className="text-sm text-muted">{what}…</p>;
}

export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: "pos" | "neg" }) {
  const color = tone === "pos" ? "text-emerald-300" : tone === "neg" ? "text-red-300" : "text-fg";
  return (
    <div className="min-w-0">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className={`truncate text-sm font-semibold tabular-nums ${color}`}>{value}</div>
    </div>
  );
}

/** Two-step in-page confirmation (no window.confirm). */
export function ConfirmButton({
  label,
  confirmLabel = "Yes, do it",
  prompt,
  onConfirm,
  disabled,
  danger,
}: {
  label: string;
  confirmLabel?: string;
  prompt: string;
  onConfirm: () => void | Promise<void>;
  disabled?: boolean;
  danger?: boolean;
}) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  if (!asking) {
    return (
      <button
        type="button"
        disabled={disabled}
        onClick={() => setAsking(true)}
        className={danger ? "btn btn-danger" : "btn"}
      >
        {label}
      </button>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-2 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1.5 text-xs">
      <span className="text-amber-200">{prompt}</span>
      <button
        type="button"
        disabled={busy}
        className="btn btn-danger"
        onClick={async () => {
          setBusy(true);
          try {
            await onConfirm();
          } finally {
            setBusy(false);
            setAsking(false);
          }
        }}
      >
        {busy ? "…" : confirmLabel}
      </button>
      <button type="button" className="btn" onClick={() => setAsking(false)} disabled={busy}>
        Cancel
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Generic JSON viewer with collapsible objects.
// ---------------------------------------------------------------------------

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function Scalar({ v }: { v: unknown }) {
  if (v === null || v === undefined) return <span className="text-slate-500">null</span>;
  if (typeof v === "boolean")
    return <span className={v ? "text-emerald-300" : "text-red-300"}>{String(v)}</span>;
  if (typeof v === "number") return <span className="text-sky-300 tabular-nums">{v}</span>;
  return <span className="break-all text-amber-100">{String(v)}</span>;
}

export function JsonView({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (Array.isArray(value)) {
    if (value.every((x) => !isObj(x) && !Array.isArray(x))) {
      return (
        <span className="font-mono text-xs">
          [
          {value.map((x, i) => (
            <span key={i}>
              {i > 0 ? ", " : ""}
              <Scalar v={x} />
            </span>
          ))}
          ]
        </span>
      );
    }
    return (
      <div className="space-y-1">
        {value.map((x, i) => (
          <JsonEntry key={i} k={`[${i}]`} v={x} depth={depth} />
        ))}
      </div>
    );
  }
  if (isObj(value)) {
    const entries = Object.entries(value);
    if (entries.length === 0) return <span className="font-mono text-xs text-slate-500">{"{}"}</span>;
    return (
      <div className="space-y-1">
        {entries.map(([k, v]) => (
          <JsonEntry key={k} k={k} v={v} depth={depth} />
        ))}
      </div>
    );
  }
  return (
    <span className="font-mono text-xs">
      <Scalar v={value} />
    </span>
  );
}

function JsonEntry({ k, v, depth }: { k: string; v: unknown; depth: number }) {
  const nested = isObj(v) || (Array.isArray(v) && v.some((x) => isObj(x) || Array.isArray(x)));
  if (!nested) {
    return (
      <div className="grid grid-cols-[minmax(8rem,auto)_1fr] gap-3 font-mono text-xs">
        <span className="text-muted">{k}</span>
        <JsonView value={v} depth={depth + 1} />
      </div>
    );
  }
  return (
    <details open={depth < 1} className="rounded border border-edge/70 bg-black/10">
      <summary className="cursor-pointer select-none px-2 py-1 font-mono text-xs text-slate-200">{k}</summary>
      <div className="border-t border-edge/70 px-3 py-2">
        <JsonView value={v} depth={depth + 1} />
      </div>
    </details>
  );
}
