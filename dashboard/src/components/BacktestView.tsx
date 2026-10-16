"use client";

import { useState } from "react";
import { api, backendUrl, errorText, isOffline } from "../lib/api";
import { useApi } from "../lib/hooks";
import { BackendOffline, ErrorNote, JsonView, Loading, Panel } from "./ui";

export function BacktestView() {
  const bt = useApi("backtest", (s) => api.backtest(s), 60_000);
  const [tab, setTab] = useState<"report" | "research">("report");

  if (!bt.data) {
    if (bt.error) return isOffline(bt.error) ? <BackendOffline /> : <ErrorNote text={errorText(bt.error)} />;
    return <Loading />;
  }
  const d = bt.data;
  const reportUrl = d.report_url ? backendUrl(d.report_url) : null;
  const researchUrl = d.research_url ? backendUrl(d.research_url) : null;
  const frameUrl = tab === "report" ? reportUrl : researchUrl;

  return (
    <div className="space-y-4">
      {isOffline(bt.error) ? <BackendOffline detail="Showing last known data." /> : null}
      <div
        className={`rounded-lg border p-4 ${
          d.gate.pass ? "border-emerald-500/50 bg-emerald-500/10" : "border-red-500/50 bg-red-500/10"
        }`}
      >
        <div className="text-xs uppercase tracking-wider text-muted">Backtest gate</div>
        <div className={`mt-1 text-2xl font-bold ${d.gate.pass ? "text-emerald-300" : "text-red-300"}`}>
          {d.gate.pass ? "PASS" : "FAIL"}
        </div>
        <p className="mt-1 whitespace-pre-wrap text-sm text-slate-200">{d.gate.text}</p>
        <p className="mt-2 text-sm">
          <span className="text-muted">Model version: </span>
          <span className="font-mono">{d.model_version ?? "none – ML skipped"}</span>
        </p>
      </div>

      <Panel title="Summary">
        <JsonView value={d.summary} />
      </Panel>

      <Panel
        title="Reports"
        right={
          <div className="flex flex-wrap gap-1">
            <button type="button" className={`btn ${tab === "report" ? "btn-on" : ""}`} onClick={() => setTab("report")}>
              Backtest report
            </button>
            <button type="button" className={`btn ${tab === "research" ? "btn-on" : ""}`} onClick={() => setTab("research")}>
              Research report
            </button>
          </div>
        }
      >
        <div className="mb-2 flex flex-wrap gap-3 text-xs">
          {reportUrl ? (
            <a className="text-sky-300 underline" href={reportUrl} target="_blank" rel="noreferrer">
              Open backtest report ↗
            </a>
          ) : (
            <span className="text-muted">No backtest report</span>
          )}
          {researchUrl ? (
            <a className="text-sky-300 underline" href={researchUrl} target="_blank" rel="noreferrer">
              Open research report ↗
            </a>
          ) : (
            <span className="text-muted">No research report</span>
          )}
        </div>
        {frameUrl ? (
          <iframe
            key={frameUrl}
            src={frameUrl}
            title={tab === "report" ? "Backtest report" : "Research report"}
            className="h-[75vh] w-full rounded border border-edge bg-white"
          />
        ) : (
          <p className="text-sm text-muted">Report not available.</p>
        )}
      </Panel>
    </div>
  );
}
