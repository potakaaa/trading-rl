"use client";

import { useState } from "react";
import { api, errorText, type StatusResponse } from "../lib/api";
import { fmtNum, fmtUsd } from "../lib/format";
import { ConfirmButton, ErrorNote, LimitBar, Panel, Stat } from "./ui";

export function RiskPanel({ status, onChanged }: { status: StatusResponse; onChanged: () => void }) {
  const r = status.risk;
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setMsg(null);
    try {
      await fn();
      setMsg({ ok: true, text: `${label} saved` });
      onChanged();
    } catch (e) {
      setMsg({ ok: false, text: `${label}: ${errorText(e)}` });
    }
  };

  return (
    <Panel
      title="Risk"
      right={
        r.halted ? (
          <span className="rounded bg-red-500/20 px-2 py-0.5 text-xs font-bold text-red-300 ring-1 ring-red-500/50">HALTED</span>
        ) : (
          <span className="text-xs text-emerald-300">not halted</span>
        )
      }
    >
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Today P&L" value={fmtUsd(r.day_pnl)} tone={r.day_pnl > 0 ? "pos" : r.day_pnl < 0 ? "neg" : undefined} />
        <Stat label="Balance" value={fmtUsd(r.balance)} />
        <Stat label="Signals today" value={`${r.signals_today} / ${r.max_signals_per_day}`} />
        <Stat
          label="Consec. losses"
          value={`${r.consecutive_losses} / ${r.max_consecutive_losses}`}
          tone={r.consecutive_losses >= r.max_consecutive_losses ? "neg" : undefined}
        />
      </div>
      <div className="mt-3 space-y-3">
        <LimitBar label="Daily loss used" used={r.daily_loss_used_usd} buffer={r.daily_buffer_usd} limit={r.daily_limit_usd} />
        <LimitBar label="Drawdown used" used={r.drawdown_used_usd} buffer={r.dd_buffer_usd} limit={r.max_dd_usd} />
        <p className="text-[11px] text-muted">
          Tick = internal buffer (signals stop there); bar end = firm limit. Initial balance {fmtUsd(r.initial_balance, 0)}.
        </p>
      </div>

      <div
        className={`mt-3 flex flex-wrap items-center gap-2 rounded border p-2 text-xs ${
          r.halted ? "border-red-500/40 bg-red-500/10" : "border-edge"
        }`}
      >
        <p className={`flex-1 ${r.halted ? "text-red-200" : "text-muted"}`}>
          {r.halted ? `Halted: ${r.halt_reason || "no reason given"}` : "No drawdown halt active."}
        </p>
        <ConfirmButton
          label="Reset halt"
          danger
          disabled={!r.halted}
          prompt="Clear the drawdown halt and allow new signals?"
          confirmLabel="Yes, reset"
          onConfirm={() => run("Halt reset", () => api.resetHalt())}
        />
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <NumberSetter
          label="Balance override ($)"
          placeholder={fmtNum(r.balance)}
          onSave={(v) => run("Balance", () => api.setBalance(v))}
        />
        <NumberSetter
          label={`Price offset (broker − feed), now ${fmtNum(status.price_offset)}`}
          placeholder={String(status.price_offset)}
          allowNegative
          onSave={(v) => run("Price offset", () => api.setPriceOffset(v))}
        />
      </div>
      {msg ? (
        <div className="mt-2">
          {msg.ok ? <p className="text-xs text-emerald-300">{msg.text}</p> : <ErrorNote text={msg.text} />}
        </div>
      ) : null}
    </Panel>
  );
}

function NumberSetter({
  label,
  placeholder,
  onSave,
  allowNegative,
}: {
  label: string;
  placeholder: string;
  onSave: (v: number) => Promise<void>;
  allowNegative?: boolean;
}) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const n = Number(value);
  const valid = value.trim() !== "" && Number.isFinite(n) && (allowNegative || n >= 0);
  return (
    <form
      className="space-y-1 text-xs"
      onSubmit={async (e) => {
        e.preventDefault();
        if (!valid) return;
        setBusy(true);
        try {
          await onSave(n);
          setValue("");
        } finally {
          setBusy(false);
        }
      }}
    >
      <span className="text-muted">{label}</span>
      <div className="flex gap-2">
        <input
          className="field"
          inputMode="decimal"
          value={value}
          placeholder={placeholder}
          onChange={(e) => setValue(e.target.value)}
        />
        <button type="submit" className="btn btn-primary" disabled={!valid || busy}>
          {busy ? "…" : "Set"}
        </button>
      </div>
    </form>
  );
}
