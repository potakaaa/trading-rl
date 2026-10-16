"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { WS_URL, type WsMessage } from "./api";

// ---------------------------------------------------------------------------
// useApi: fetch on mount / key change, optional polling, manual reload.
// ---------------------------------------------------------------------------

export interface ApiState<T> {
  data: T | undefined;
  error: unknown;
  /** epoch ms of last successful load */
  loadedAt: number | undefined;
  reload: () => void;
  /** Locally replace data (e.g. from a WebSocket push or POST response). */
  setData: (updater: (prev: T | undefined) => T | undefined) => void;
}

export function useApi<T>(
  key: string,
  fetcher: (signal: AbortSignal) => Promise<T>,
  intervalMs?: number,
): ApiState<T> {
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });
  const [state, setState] = useState<{ data?: T; error?: unknown; loadedAt?: number }>({});
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = async () => {
      try {
        const data = await fetcherRef.current(ctrl.signal);
        if (!ctrl.signal.aborted) setState({ data, error: undefined, loadedAt: Date.now() });
      } catch (e) {
        if (!ctrl.signal.aborted) setState((s) => ({ ...s, error: e }));
      }
      if (intervalMs && !ctrl.signal.aborted) timer = setTimeout(run, intervalMs);
    };
    void run();
    return () => {
      ctrl.abort();
      if (timer) clearTimeout(timer);
    };
  }, [key, tick, intervalMs]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  const setData = useCallback(
    (updater: (prev: T | undefined) => T | undefined) =>
      setState((s) => ({ ...s, data: updater(s.data) })),
    [],
  );

  return { data: state.data, error: state.error, loadedAt: state.loadedAt, reload, setData };
}

// ---------------------------------------------------------------------------
// useNow: ticking clock (null during SSR/prerender so output is deterministic).
// ---------------------------------------------------------------------------

function subscribeClock(cb: () => void): () => void {
  const id = setInterval(cb, 1000);
  return () => clearInterval(id);
}
const clientNow = () => Math.floor(Date.now() / 1000) * 1000;
const serverNow = () => null;

export function useNow(): number | null {
  return useSyncExternalStore(subscribeClock, clientNow, serverNow);
}

// ---------------------------------------------------------------------------
// useLiveSocket: WebSocket with exponential backoff reconnect.
// ---------------------------------------------------------------------------

export function useLiveSocket(onMessage: (m: WsMessage) => void): boolean {
  const handler = useRef(onMessage);
  useEffect(() => {
    handler.current = onMessage;
  });
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let attempt = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let disposed = false;

    const schedule = () => {
      if (disposed) return;
      const delay = Math.min(30_000, 1000 * 2 ** attempt) + Math.random() * 500;
      attempt += 1;
      timer = setTimeout(connect, delay);
    };

    const connect = () => {
      if (disposed) return;
      try {
        ws = new WebSocket(WS_URL);
      } catch {
        schedule();
        return;
      }
      ws.onopen = () => {
        attempt = 0;
        setConnected(true);
      };
      ws.onmessage = (ev: MessageEvent) => {
        try {
          const msg = JSON.parse(typeof ev.data === "string" ? ev.data : "") as WsMessage;
          if (msg && typeof msg.type === "string") handler.current(msg);
        } catch {
          // ignore malformed frames
        }
      };
      ws.onerror = () => {
        ws?.close();
      };
      ws.onclose = () => {
        setConnected(false);
        schedule();
      };
    };

    connect();
    return () => {
      disposed = true;
      if (timer) clearTimeout(timer);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, []);

  return connected;
}

// ---------------------------------------------------------------------------
// Tiny pub/sub so the chart can receive live bars without prop churn.
// ---------------------------------------------------------------------------

export function createBus<T>() {
  const subs = new Set<(v: T) => void>();
  return {
    emit(v: T) {
      subs.forEach((fn) => fn(v));
    },
    on(fn: (v: T) => void): () => void {
      subs.add(fn);
      return () => {
        subs.delete(fn);
      };
    },
  };
}
