// Typed client for the FastAPI live backend. Mirrors docs/API.md.
// All timestamps are ISO-8601 UTC strings; prices are feed index points.

export const API_BASE: string = (
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

/** WebSocket endpoint derived from the REST base (http -> ws, https -> wss). */
export const WS_URL = `${API_BASE.replace(/^http/, "ws")}/ws`;

/** Build an absolute URL for a backend-relative path such as `/reports/x.html`. */
export function backendUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Mode = "paper" | "live";

export type EngineStatusCode =
  | "ACTIVE"
  | "PAUSED-NEWS"
  | "STOPPED-FOR-DAY"
  | "HALTED"
  | "NO-DATA"
  | "NO-NEWS"
  | "OUTSIDE-SESSION";

export type Direction = "long" | "short";
export type Timeframe = "1m" | "5m" | "15m";
export const TIMEFRAMES: Timeframe[] = ["1m", "5m", "15m"];

export interface EngineInfo {
  running: boolean;
  started_at: string | null;
  last_bar_at: string | null;
  feed_ok: boolean;
  feed_lag_seconds: number | null;
  feed_error: string | null;
}

export interface RiskState {
  initial_balance: number;
  balance: number;
  day_pnl: number;
  daily_limit_usd: number;
  daily_buffer_usd: number;
  daily_loss_used_usd: number;
  max_dd_usd: number;
  dd_buffer_usd: number;
  drawdown_used_usd: number;
  signals_today: number;
  max_signals_per_day: number;
  consecutive_losses: number;
  max_consecutive_losses: number;
  halted: boolean;
  halt_reason: string;
}

export interface NextNewsEvent {
  title: string;
  ts: string;
}

export interface NewsStatus {
  available: boolean;
  blackout: boolean;
  reason: string;
  next_event: NextNewsEvent | null;
  minutes_to_next: number | null;
  last_refresh: string | null;
}

export interface StatusResponse {
  mode: Mode;
  engine: EngineInfo;
  status: EngineStatusCode;
  status_reason: string;
  session: string | null;
  price: number | null;
  price_offset: number;
  risk: RiskState;
  news: NewsStatus;
  warnings: string[];
}

export interface Bar {
  /** bar start, ISO UTC */
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
}

export interface BarsResponse {
  tf: Timeframe;
  bars: Bar[];
}

export interface LiquidityLevel {
  kind: string;
  price: number;
  side: "high" | "low";
  known_at: string;
  consumed_at: string | null;
}

export interface Fvg {
  direction: "bull" | "bear";
  top: number;
  bottom: number;
  start: string;
  end: string | null;
  filled_at: string | null;
}

export interface OrderBlock {
  top: number;
  bottom: number;
  start: string;
  direction: "bull" | "bear";
}

export interface Sweep {
  t: string;
  price: number;
  level_kind: string;
  direction: Direction;
}

export interface Mss {
  t: string;
  price: number;
  direction: Direction;
}

export interface Swing {
  t: string;
  price: number;
  kind: "high" | "low";
}

export interface OverlaysResponse {
  levels: LiquidityLevel[];
  fvgs: Fvg[];
  order_blocks: OrderBlock[];
  sweeps: Sweep[];
  mss: Mss[];
  swings: Swing[];
}

export type SignalStatus = "active" | "expired" | "filled" | "closed" | "cancelled";
export type SignalAction = "taken" | "skipped";

export interface PaperResult {
  outcome: string | null;
  fill_time: string | null;
  exit_time: string | null;
  r: number | null;
}

export interface Signal {
  id: number;
  created_at: string;
  setup_type: string;
  direction: Direction;
  session: string | null;
  entry: number;
  stop: number;
  target: number;
  rr: number;
  risk_pts: number;
  entry_broker: number;
  stop_broker: number;
  target_broker: number;
  lots: number;
  risk_usd: number;
  expires_at: string;
  confidence: number | null;
  confidence_note: string;
  reason: string;
  status: SignalStatus;
  paper: PaperResult;
  action: SignalAction | null;
  fill_price: number | null;
  exit_price: number | null;
  pnl_usd: number | null;
  notes: string;
}

export interface SignalsResponse {
  signals: Signal[];
}

export interface SignalActionRequest {
  action: SignalAction;
  fill_price?: number;
  exit_price?: number;
  lots?: number;
  notes?: string;
}

export interface BlockEntry {
  time: string;
  setup_type: string;
  direction: string;
  reasons: string[];
}

export interface BlocksResponse {
  blocks: BlockEntry[];
}

export interface NewsEvent {
  ts: string;
  title: string;
  blackout_start: string;
  blackout_end: string;
}

export interface NewsResponse {
  available: boolean;
  last_refresh: string | null;
  events: NewsEvent[];
}

export interface JournalStats {
  taken: number;
  wins: number;
  losses: number;
  total_r: number;
  total_pnl_usd: number;
  paper_total_r: number;
  paper_trades: number;
}

export interface JournalResponse {
  signals: Signal[];
  stats: JournalStats;
}

export interface JournalQuery {
  from?: string;
  to?: string;
  setup_type?: string;
  action?: string;
}

export interface BacktestGate {
  pass: boolean;
  text: string;
}

export interface BacktestResponse {
  report_url: string | null;
  research_url: string | null;
  summary: Record<string, unknown>;
  gate: BacktestGate;
  model_version: string | null;
}

export type SettingsResponse = Record<string, unknown>;

export type AlertLevel = "info" | "warning" | "critical";

export interface AlertPayload {
  level: AlertLevel;
  text: string;
}

export interface BarMessagePayload {
  tf: "1m";
  bar: Bar;
}

export type WsMessage =
  | { type: "status"; data: StatusResponse }
  | { type: "bar"; data: BarMessagePayload }
  | { type: "signal"; data: Signal }
  | { type: "block"; data: BlockEntry }
  | { type: "alert"; data: AlertPayload };

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Thrown when the backend cannot be reached at all (connection refused, timeout, CORS). */
export class OfflineError extends Error {
  constructor(message = "Backend unreachable") {
    super(message);
    this.name = "OfflineError";
  }
}

export function isOffline(err: unknown): boolean {
  return err instanceof OfflineError;
}

export function errorText(err: unknown): string {
  if (err instanceof ApiError) return `HTTP ${err.status}: ${err.message}`;
  if (err instanceof Error) return err.message;
  return String(err);
}

const TIMEOUT_MS = 10_000;

async function request<T>(
  path: string,
  init: RequestInit & { signal?: AbortSignal } = {},
): Promise<T> {
  const timeout = AbortSignal.timeout(TIMEOUT_MS);
  const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal,
      cache: "no-store",
      headers: {
        Accept: "application/json",
        ...(init.body ? { "Content-Type": "application/json" } : {}),
        ...init.headers,
      },
    });
  } catch (e) {
    if (init.signal?.aborted) throw e;
    throw new OfflineError(e instanceof Error ? e.message : undefined);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (body && body.detail !== undefined) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // non-JSON error body
    }
    throw new ApiError(res.status, detail || "request failed");
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, string | number | undefined | null>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const api = {
  status: (signal?: AbortSignal) => request<StatusResponse>("/api/status", { signal }),

  bars: (tf: Timeframe, limit = 500, signal?: AbortSignal) =>
    request<BarsResponse>(`/api/bars${qs({ tf, limit })}`, { signal }),

  overlays: (day?: string, signal?: AbortSignal) =>
    request<OverlaysResponse>(`/api/overlays${qs({ day })}`, { signal }),

  signals: (limit = 50, signal?: AbortSignal) =>
    request<SignalsResponse>(`/api/signals${qs({ limit })}`, { signal }),

  signalAction: (id: number, body: SignalActionRequest) =>
    post<Signal>(`/api/signals/${id}/action`, body),

  blocks: (day?: string, signal?: AbortSignal) =>
    request<BlocksResponse>(`/api/blocks${qs({ day })}`, { signal }),

  news: (signal?: AbortSignal) => request<NewsResponse>("/api/news", { signal }),

  journal: (q: JournalQuery, signal?: AbortSignal) =>
    request<JournalResponse>(`/api/journal${qs({ ...q })}`, { signal }),

  backtest: (signal?: AbortSignal) => request<BacktestResponse>("/api/backtest", { signal }),

  settings: (signal?: AbortSignal) => request<SettingsResponse>("/api/settings", { signal }),

  setBalance: (balance: number) => post<unknown>("/api/account/balance", { balance }),

  resetHalt: () => post<unknown>("/api/account/reset", {}),

  setPriceOffset: (offset: number) => post<unknown>("/api/price-offset", { offset }),
};
