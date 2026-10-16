"use client";

import { useCallback, useRef, useState } from "react";
import { API_BASE, api, errorText, isOffline, type AlertPayload, type Signal, type WsMessage } from "../lib/api";
import { useApi, useLiveSocket, useNow } from "../lib/hooks";
import { liveBars } from "../lib/live";
import { AlertFeed, BlocksPanel, Toasts, type AlertItem } from "./BlocksAlerts";
import { ChartPanel } from "./ChartPanel";
import { NewsPanel } from "./NewsPanel";
import { RiskPanel } from "./RiskPanel";
import { SignalPanel } from "./SignalPanel";
import { StatusBar } from "./StatusBar";
import { BackendOffline, ErrorNote, Loading } from "./ui";

const TOAST_MS = 8000;

function upsertSignal(list: Signal[], s: Signal): Signal[] {
  const i = list.findIndex((x) => x.id === s.id);
  if (i === -1) return [s, ...list].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const next = list.slice();
  next[i] = s;
  return next;
}

export function LiveDashboard() {
  const now = useNow();
  const status = useApi("status", (s) => api.status(s), 15_000);
  const signals = useApi("signals", (s) => api.signals(50, s), 30_000);
  const blocks = useApi("blocks", (s) => api.blocks(undefined, s), 60_000);
  const news = useApi("news", (s) => api.news(s), 300_000);

  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [toasts, setToasts] = useState<AlertItem[]>([]);
  const alertId = useRef(0);

  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);

  const pushAlert = useCallback(
    (a: AlertPayload) => {
      alertId.current += 1;
      const item: AlertItem = { id: alertId.current, level: a.level, text: a.text, at: Date.now() };
      setAlerts((prev) => [item, ...prev].slice(0, 100));
      setToasts((prev) => [...prev, item].slice(-5));
      if (a.level !== "critical") setTimeout(() => dismiss(item.id), TOAST_MS);
    },
    [dismiss],
  );

  const setStatus = status.setData;
  const setSignals = signals.setData;
  const setBlocks = blocks.setData;
  const onSignal = useCallback(
    (s: Signal) => setSignals((prev) => ({ signals: upsertSignal(prev?.signals ?? [], s) })),
    [setSignals],
  );

  const wsConnected = useLiveSocket((msg: WsMessage) => {
    switch (msg.type) {
      case "status":
        setStatus(() => msg.data);
        break;
      case "bar":
        if (msg.data?.bar) liveBars.emit(msg.data.bar);
        break;
      case "signal":
        onSignal(msg.data);
        break;
      case "block":
        setBlocks((prev) => ({ blocks: [msg.data, ...(prev?.blocks ?? [])] }));
        break;
      case "alert":
        pushAlert(msg.data);
        break;
    }
  });

  const offline = isOffline(status.error);

  if (!status.data) {
    return (
      <div className="space-y-4">
        {status.error ? (
          offline ? (
            <BackendOffline detail={`Could not reach ${API_BASE} (${errorText(status.error)})`} />
          ) : (
            <ErrorNote text={`Status: ${errorText(status.error)}`} />
          )
        ) : (
          <Loading what="Connecting to backend" />
        )}
        <Toasts toasts={toasts} onDismiss={dismiss} />
      </div>
    );
  }

  const st = status.data;
  return (
    <div className="space-y-4">
      {offline ? <BackendOffline detail="Showing last known data." /> : null}
      <StatusBar status={st} wsConnected={wsConnected} now={now} />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="min-w-0 space-y-4 xl:col-span-2">
          <ChartPanel signals={signals.data?.signals ?? []} />
          <div className="grid gap-4 lg:grid-cols-2">
            <RiskPanel status={st} onChanged={status.reload} />
            <NewsPanel news={news.data} error={news.error} live={st.news} now={now} />
          </div>
        </div>
        <div className="min-w-0 space-y-4">
          {signals.error && !signals.data ? <ErrorNote text={`Signals: ${errorText(signals.error)}`} /> : null}
          <SignalPanel signals={signals.data?.signals ?? []} now={now} onUpdated={(s) => { onSignal(s); status.reload(); }} />
          <BlocksPanel blocks={blocks.data?.blocks} error={blocks.error && !blocks.data ? errorText(blocks.error) : undefined} />
          <AlertFeed alerts={alerts} />
        </div>
      </div>
      <Toasts toasts={toasts} onDismiss={dismiss} />
    </div>
  );
}
