"use client";

import { useState } from "react";
import { api, errorText, type Signal, type SignalActionRequest } from "../lib/api";
import { TZ_MANILA, TZ_NY, fmtCountdown, fmtDateTime, fmtNum, fmtSigned, fmtTime, fmtUsd } from "../lib/format";
import { DirectionPill, ErrorNote, Panel, Pill, type Tone } from "./ui";

const STATUS_TONE: Record<string, Tone> = {
  active: "blue",
  filled: "violet",
  closed: "slate",
  expired: "slate",
  cancelled: "slate",
};

export function SignalPanel({
  signals,
  now,
  onUpdated,
}: {
  signals: Signal[];
  now: number | null;
  onUpdated: (s: Signal) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const active = signals.filter((s) => s.status === "active");
  const recent = signals.filter((s) => s.status !== "active");
  const recentShown = showAll ? recent : recent.slice(0, 5);
  return (
    <Panel title={`Signals · ${active.length} active`}>
      {signals.length === 0 ? <p className="text-sm text-muted">No signals yet.</p> : null}
      <div className="space-y-3">
        {active.map((s) => (
          <SignalCard key={s.id} signal={s} now={now} onUpdated={onUpdated} />
        ))}
        {recent.length ? (
          <h3 className="pt-1 text-[11px] uppercase tracking-wider text-muted">Recent</h3>
        ) : null}
        {recentShown.map((s) => (
          <SignalCard key={s.id} signal={s} now={now} onUpdated={onUpdated} />
        ))}
        {recent.length > 5 ? (
          <button type="button" className="btn w-full" onClick={() => setShowAll((v) => !v)}>
            {showAll ? "Show fewer" : `Show ${recent.length - 5} more`}
          </button>
        ) : null}
      </div>
    </Panel>
  );
}

function PriceRow({ label, feed, broker, color }: { label: string; feed: number; broker: number; color: string }) {
  return (
    <>
      <span className={`text-muted ${color}`}>{label}</span>
      <span className="text-right tabular-nums">{fmtNum(feed)}</span>
      <span className="text-right tabular-nums text-slate-400">{fmtNum(broker)}</span>
    </>
  );
}

function SignalCard({ signal: s, now, onUpdated }: { signal: Signal; now: number | null; onUpdated: (s: Signal) => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const isActive = s.status === "active";
  const expiresMs = now === null ? null : new Date(s.expires_at).getTime() - now;

  const send = async (body: SignalActionRequest) => {
    setBusy(true);
    setErr(null);
    try {
      const updated = await api.signalAction(s.id, body);
      onUpdated(updated);
      return true;
    } catch (e) {
      setErr(errorText(e));
      return false;
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className={`rounded-md border p-3 ${isActive ? "border-sky-500/40 bg-sky-500/[0.04]" : "border-edge bg-black/10"}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs text-muted">#{s.id}</span>
        <DirectionPill direction={s.direction} />
        <span className="text-sm font-semibold">{s.setup_type}</span>
        <Pill tone={STATUS_TONE[s.status] ?? "slate"}>{s.status}</Pill>
        {s.session ? <Pill>{s.session}</Pill> : null}
        {s.action ? <Pill tone={s.action === "taken" ? "green" : "amber"}>{s.action}</Pill> : null}
      </div>
      <p className="mt-1 text-[11px] text-muted">
        NY {fmtDateTime(s.created_at, TZ_NY)} · MNL {fmtTime(s.created_at, TZ_MANILA)}
        {isActive && expiresMs !== null ? (
          <span className={`ml-2 font-semibold ${expiresMs < 120_000 ? "text-amber-300" : "text-sky-300"}`}>
            {expiresMs > 0 ? `expires in ${fmtCountdown(expiresMs)}` : "expiring…"}
          </span>
        ) : null}
      </p>

      <div className="mt-2 grid grid-cols-[auto_1fr_1fr] gap-x-3 gap-y-0.5 text-xs">
        <span />
        <span className="text-right text-[10px] uppercase text-muted">feed</span>
        <span className="text-right text-[10px] uppercase text-muted">broker</span>
        <PriceRow label="Entry" feed={s.entry} broker={s.entry_broker} color="text-sky-300" />
        <PriceRow label="SL" feed={s.stop} broker={s.stop_broker} color="text-red-300" />
        <PriceRow label="TP" feed={s.target} broker={s.target_broker} color="text-emerald-300" />
      </div>
      <div className="mt-2 grid grid-cols-4 gap-2 text-xs">
        <Kv k="R:R" v={fmtNum(s.rr, 2)} />
        <Kv k="Risk pts" v={fmtNum(s.risk_pts, 1)} />
        <Kv k="Lots" v={fmtNum(s.lots, 2)} />
        <Kv k="Risk $" v={fmtUsd(s.risk_usd)} />
      </div>
      <p className="mt-2 text-xs text-slate-300">{s.reason}</p>
      <p className="mt-1 text-[11px] italic text-muted">
        {s.confidence !== null ? `confidence ${(s.confidence * 100).toFixed(0)}% · ` : ""}
        {s.confidence_note}
      </p>
      <p className="mt-1 text-[11px] text-muted">
        Paper:{" "}
        {s.paper.outcome ? (
          <span className="text-slate-200">
            {s.paper.outcome} {s.paper.r !== null ? `(${fmtSigned(s.paper.r, 2, "R")})` : ""}
            {s.paper.fill_time ? ` · fill ${fmtTime(s.paper.fill_time, TZ_NY)}` : ""}
            {s.paper.exit_time ? ` · exit ${fmtTime(s.paper.exit_time, TZ_NY)} NY` : ""}
          </span>
        ) : (
          <span>{s.paper.fill_time ? `filled ${fmtTime(s.paper.fill_time, TZ_NY)} NY, open` : "pending"}</span>
        )}
      </p>
      {s.action === "taken" && (s.fill_price !== null || s.pnl_usd !== null) ? (
        <p className="mt-1 text-[11px] text-slate-300">
          You: fill {fmtNum(s.fill_price)} · exit {fmtNum(s.exit_price)} · P&L{" "}
          <span className={(s.pnl_usd ?? 0) >= 0 ? "text-emerald-300" : "text-red-300"}>{fmtUsd(s.pnl_usd)}</span>
          {s.notes ? ` · ${s.notes}` : ""}
        </p>
      ) : null}

      <div className="mt-2 flex flex-wrap gap-2">
        <button
          type="button"
          disabled={busy}
          className={`btn ${s.action === "taken" ? "btn-on" : ""}`}
          onClick={async () => {
            if (s.action !== "taken") {
              const ok = await send({ action: "taken" });
              if (ok) setFormOpen(true);
            } else {
              setFormOpen((v) => !v);
            }
          }}
        >
          {s.action === "taken" ? (formOpen ? "Hide fill form" : "Edit fill / exit") : "Taken"}
        </button>
        <button
          type="button"
          disabled={busy || s.action === "skipped"}
          className={`btn ${s.action === "skipped" ? "btn-on" : ""}`}
          onClick={async () => {
            const ok = await send({ action: "skipped" });
            if (ok) setFormOpen(false);
          }}
        >
          Skipped
        </button>
      </div>
      {err ? (
        <div className="mt-2">
          <ErrorNote text={err} />
        </div>
      ) : null}
      {formOpen && s.action === "taken" ? <FillForm signal={s} busy={busy} onSubmit={send} /> : null}
    </article>
  );
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase text-muted">{k}</div>
      <div className="font-semibold tabular-nums">{v}</div>
    </div>
  );
}

function parseNum(v: string): number | undefined {
  if (v.trim() === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function FillForm({
  signal: s,
  busy,
  onSubmit,
}: {
  signal: Signal;
  busy: boolean;
  onSubmit: (b: SignalActionRequest) => Promise<boolean>;
}) {
  const [fill, setFill] = useState(s.fill_price !== null ? String(s.fill_price) : "");
  const [exit, setExit] = useState(s.exit_price !== null ? String(s.exit_price) : "");
  const [lots, setLots] = useState(String(s.lots));
  const [notes, setNotes] = useState(s.notes ?? "");
  const [saved, setSaved] = useState(false);
  return (
    <form
      className="mt-2 grid grid-cols-2 gap-2 rounded border border-edge bg-black/20 p-2 text-xs sm:grid-cols-4"
      onSubmit={async (e) => {
        e.preventDefault();
        setSaved(false);
        const body: SignalActionRequest = { action: "taken", notes };
        const f = parseNum(fill);
        const x = parseNum(exit);
        const l = parseNum(lots);
        if (f !== undefined) body.fill_price = f;
        if (x !== undefined) body.exit_price = x;
        if (l !== undefined) body.lots = l;
        setSaved(await onSubmit(body));
      }}
    >
      <label className="space-y-0.5">
        <span className="text-muted">Fill price</span>
        <input className="field" inputMode="decimal" value={fill} onChange={(e) => setFill(e.target.value)} placeholder={fmtNum(s.entry_broker)} />
      </label>
      <label className="space-y-0.5">
        <span className="text-muted">Exit</span>
        <input className="field" inputMode="decimal" value={exit} onChange={(e) => setExit(e.target.value)} placeholder="open" />
      </label>
      <label className="space-y-0.5">
        <span className="text-muted">Lots</span>
        <input className="field" inputMode="decimal" value={lots} onChange={(e) => setLots(e.target.value)} />
      </label>
      <label className="col-span-2 space-y-0.5 sm:col-span-1">
        <span className="text-muted">Notes</span>
        <input className="field" value={notes} onChange={(e) => setNotes(e.target.value)} />
      </label>
      <div className="col-span-2 flex items-center gap-2 sm:col-span-4">
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? "Saving…" : "Save"}
        </button>
        {saved ? <span className="text-emerald-300">saved</span> : null}
      </div>
    </form>
  );
}
