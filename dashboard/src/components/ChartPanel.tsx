"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { TIMEFRAMES, api, errorText, type Bar, type Signal, type Timeframe } from "../lib/api";
import { mergeMinuteBar } from "../lib/bars";
import { useApi } from "../lib/hooks";
import { liveBars } from "../lib/live";
import { ErrorNote, Panel } from "./ui";

const CandleChart = dynamic(() => import("./CandleChart"), {
  ssr: false,
  loading: () => <div className="h-[340px] w-full animate-pulse rounded bg-white/[0.02] sm:h-[440px] xl:h-[520px]" />,
});

interface BarsState {
  tf: Timeframe;
  bars: Bar[];
  error?: string;
}

export function ChartPanel({ signals }: { signals: Signal[] }) {
  const [tf, setTf] = useState<Timeframe>("5m");
  const [state, setState] = useState<BarsState | null>(null);
  const [showSwings, setShowSwings] = useState(false);
  const overlays = useApi("overlays", (s) => api.overlays(undefined, s), 60_000);
  const reloadOverlays = overlays.reload;

  // Fetch bars whenever the timeframe changes (and re-sync every 5 minutes).
  useEffect(() => {
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      try {
        const res = await api.bars(tf, 500, ctrl.signal);
        if (!ctrl.signal.aborted) setState({ tf, bars: res.bars });
      } catch (e) {
        if (!ctrl.signal.aborted)
          setState((prev) => ({ tf, bars: prev?.tf === tf ? prev.bars : [], error: errorText(e) }));
      }
      if (!ctrl.signal.aborted) timer = setTimeout(load, 300_000);
    };
    void load();
    return () => {
      ctrl.abort();
      if (timer) clearTimeout(timer);
    };
  }, [tf]);

  // Live 1m bars from the WebSocket.
  useEffect(
    () =>
      liveBars.on((bar) => {
        setState((prev) => (prev ? { ...prev, bars: mergeMinuteBar(prev.bars, bar, prev.tf), error: undefined } : prev));
        reloadOverlays();
      }),
    [reloadOverlays],
  );

  const bars = state && state.tf === tf ? state.bars : [];
  const loading = !state || state.tf !== tf;

  return (
    <Panel
      title="US100"
      right={
        <div className="flex items-center gap-1">
          <label className="mr-2 flex items-center gap-1 text-[11px] text-muted">
            <input type="checkbox" checked={showSwings} onChange={(e) => setShowSwings(e.target.checked)} />
            swings
          </label>
          {TIMEFRAMES.map((t) => (
            <button key={t} type="button" onClick={() => setTf(t)} className={`btn ${t === tf ? "btn-on" : ""}`}>
              {t}
            </button>
          ))}
        </div>
      }
    >
      <div className="relative">
        <CandleChart tf={tf} bars={bars} overlays={overlays.data ?? null} signals={signals} showSwings={showSwings} />
        {loading ? (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-muted">
            Loading {tf} bars…
          </div>
        ) : bars.length === 0 ? (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-muted">
            No bars available
          </div>
        ) : null}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted">
        <span>Times in New York</span>
        <Legend color="#f472b6" label="high-side liquidity" dashed />
        <Legend color="#38bdf8" label="low-side liquidity" dashed />
        <Legend color="rgba(34,197,94,0.6)" label="FVG" box />
        <Legend color="rgba(56,189,248,0.6)" label="OB" box />
        <Legend color="#60a5fa" label="entry" />
        <Legend color="#f87171" label="SL" />
        <Legend color="#4ade80" label="TP" />
      </div>
      {state?.error ? (
        <div className="mt-2">
          <ErrorNote text={`bars: ${state.error}`} />
        </div>
      ) : null}
      {overlays.error ? (
        <div className="mt-2">
          <ErrorNote text={`overlays: ${errorText(overlays.error)}`} />
        </div>
      ) : null}
    </Panel>
  );
}

function Legend({ color, label, dashed, box }: { color: string; label: string; dashed?: boolean; box?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1">
      {box ? (
        <span className="inline-block h-2.5 w-3.5 rounded-sm border" style={{ borderColor: color, background: `${color.replace(/[\d.]+\)$/, "0.15)")}` }} />
      ) : (
        <span className="inline-block w-4 border-t-2" style={{ borderColor: color, borderTopStyle: dashed ? "dashed" : "solid" }} />
      )}
      {label}
    </span>
  );
}
