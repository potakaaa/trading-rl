"use client";

import { useState } from "react";
import { api, errorText, isOffline, type JournalQuery } from "../lib/api";
import { TZ_MANILA, TZ_NY, fmtDateTime, fmtNum, fmtSigned, fmtTime, fmtUsd } from "../lib/format";
import { useApi } from "../lib/hooks";
import { BackendOffline, DirectionPill, ErrorNote, Loading, Panel, Pill, Stat } from "./ui";

const EMPTY: JournalQuery = { from: "", to: "", setup_type: "", action: "" };

export function JournalView() {
  const [draft, setDraft] = useState<JournalQuery>(EMPTY);
  const [query, setQuery] = useState<JournalQuery>(EMPTY);
  const journal = useApi(`journal:${JSON.stringify(query)}`, (s) => api.journal(query, s));
  const setupTypes = Array.from(new Set((journal.data?.signals ?? []).map((s) => s.setup_type))).sort();
  const st = journal.data?.stats;

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Journal</h1>
      <form
        className="grid grid-cols-2 gap-2 rounded-lg border border-edge bg-panel p-3 text-xs sm:grid-cols-5"
        onSubmit={(e) => {
          e.preventDefault();
          setQuery(draft);
        }}
      >
        <label className="space-y-0.5">
          <span className="text-muted">From</span>
          <input type="date" className="field" value={draft.from} onChange={(e) => setDraft({ ...draft, from: e.target.value })} />
        </label>
        <label className="space-y-0.5">
          <span className="text-muted">To</span>
          <input type="date" className="field" value={draft.to} onChange={(e) => setDraft({ ...draft, to: e.target.value })} />
        </label>
        <label className="space-y-0.5">
          <span className="text-muted">Setup type</span>
          <input
            className="field"
            list="setup-types"
            placeholder="any"
            value={draft.setup_type}
            onChange={(e) => setDraft({ ...draft, setup_type: e.target.value })}
          />
          <datalist id="setup-types">
            {setupTypes.map((t) => (
              <option key={t} value={t} />
            ))}
          </datalist>
        </label>
        <label className="space-y-0.5">
          <span className="text-muted">Action</span>
          <select className="field" value={draft.action} onChange={(e) => setDraft({ ...draft, action: e.target.value })}>
            <option value="">any</option>
            <option value="taken">taken</option>
            <option value="skipped">skipped</option>
          </select>
        </label>
        <div className="col-span-2 flex items-end gap-2 sm:col-span-1">
          <button type="submit" className="btn btn-primary">
            Apply
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              setDraft(EMPTY);
              setQuery(EMPTY);
            }}
          >
            Clear
          </button>
        </div>
      </form>

      {journal.error ? (
        isOffline(journal.error) ? <BackendOffline /> : <ErrorNote text={errorText(journal.error)} />
      ) : null}
      {!journal.data && !journal.error ? <Loading /> : null}

      {st ? (
        <div className="grid grid-cols-2 gap-3 rounded-lg border border-edge bg-panel p-3 sm:grid-cols-4 lg:grid-cols-7">
          <Stat label="Taken" value={st.taken} />
          <Stat label="Wins" value={st.wins} tone="pos" />
          <Stat label="Losses" value={st.losses} tone="neg" />
          <Stat label="Total R" value={fmtSigned(st.total_r, 2, "R")} tone={st.total_r > 0 ? "pos" : st.total_r < 0 ? "neg" : undefined} />
          <Stat label="Total P&L" value={fmtUsd(st.total_pnl_usd)} tone={st.total_pnl_usd > 0 ? "pos" : st.total_pnl_usd < 0 ? "neg" : undefined} />
          <Stat label="Paper trades" value={st.paper_trades} />
          <Stat label="Paper total R" value={fmtSigned(st.paper_total_r, 2, "R")} tone={st.paper_total_r > 0 ? "pos" : st.paper_total_r < 0 ? "neg" : undefined} />
        </div>
      ) : null}

      {journal.data ? (
        <Panel title={`${journal.data.signals.length} signals`}>
          {journal.data.signals.length === 0 ? (
            <p className="text-sm text-muted">No signals match these filters.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1100px] text-xs">
                <thead>
                  <tr className="text-left text-[10px] uppercase tracking-wider text-muted">
                    {[
                      "#",
                      "Created (NY)",
                      "MNL",
                      "Setup",
                      "Dir",
                      "Session",
                      "Entry / SL / TP",
                      "R:R",
                      "Lots",
                      "Risk $",
                      "Status",
                      "Paper",
                      "Action",
                      "Fill → Exit",
                      "P&L",
                      "Notes",
                    ].map((h) => (
                      <th key={h} className="whitespace-nowrap py-1 pr-3 font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {journal.data.signals.map((s) => (
                    <tr key={s.id} className="border-t border-edge/60 align-top">
                      <td className="py-1.5 pr-3 font-mono text-muted">{s.id}</td>
                      <td className="whitespace-nowrap py-1.5 pr-3 tabular-nums">{fmtDateTime(s.created_at, TZ_NY)}</td>
                      <td className="whitespace-nowrap py-1.5 pr-3 tabular-nums text-muted">{fmtTime(s.created_at, TZ_MANILA)}</td>
                      <td className="py-1.5 pr-3">{s.setup_type}</td>
                      <td className="py-1.5 pr-3">
                        <DirectionPill direction={s.direction} />
                      </td>
                      <td className="py-1.5 pr-3">{s.session ?? "–"}</td>
                      <td className="whitespace-nowrap py-1.5 pr-3 tabular-nums">
                        {fmtNum(s.entry)} / <span className="text-red-300">{fmtNum(s.stop)}</span> /{" "}
                        <span className="text-emerald-300">{fmtNum(s.target)}</span>
                      </td>
                      <td className="py-1.5 pr-3 tabular-nums">{fmtNum(s.rr)}</td>
                      <td className="py-1.5 pr-3 tabular-nums">{fmtNum(s.lots)}</td>
                      <td className="py-1.5 pr-3 tabular-nums">{fmtUsd(s.risk_usd)}</td>
                      <td className="py-1.5 pr-3">{s.status}</td>
                      <td className="whitespace-nowrap py-1.5 pr-3">
                        {s.paper.outcome ?? "–"}
                        {s.paper.r !== null ? ` (${fmtSigned(s.paper.r, 2, "R")})` : ""}
                      </td>
                      <td className="py-1.5 pr-3">
                        {s.action ? <Pill tone={s.action === "taken" ? "green" : "amber"}>{s.action}</Pill> : <span className="text-muted">–</span>}
                      </td>
                      <td className="whitespace-nowrap py-1.5 pr-3 tabular-nums">
                        {s.fill_price !== null ? `${fmtNum(s.fill_price)} → ${fmtNum(s.exit_price)}` : "–"}
                      </td>
                      <td
                        className={`py-1.5 pr-3 tabular-nums ${
                          s.pnl_usd === null ? "" : s.pnl_usd >= 0 ? "text-emerald-300" : "text-red-300"
                        }`}
                      >
                        {fmtUsd(s.pnl_usd)}
                      </td>
                      <td className="max-w-[16rem] py-1.5 pr-3 text-slate-300">{s.notes || ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      ) : null}
    </div>
  );
}
