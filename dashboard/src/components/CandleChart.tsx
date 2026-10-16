"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineStyle,
  TickMarkType,
  createChart,
  createSeriesMarkers,
  type CandlestickData,
  type IChartApi,
  type IPrimitivePaneRenderer,
  type IPrimitivePaneView,
  type IPriceLine,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type ISeriesPrimitive,
  type SeriesAttachedParameter,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import type { Bar, OverlaysResponse, Signal, Timeframe } from "../lib/api";
import { bucketStart } from "../lib/bars";
import { TZ_NY, isoToSec } from "../lib/format";

// ---------------------------------------------------------------------------
// Shaded zone primitive (FVGs / order blocks)
// ---------------------------------------------------------------------------

type CanvasRenderingTarget2D = Parameters<IPrimitivePaneRenderer["draw"]>[0];

interface Zone {
  from: number; // unix sec (bucketed)
  to: number | null; // null = extend to right edge
  top: number;
  bottom: number;
  fill: string;
  stroke: string;
  label: string;
}

class ZonesPrimitive implements ISeriesPrimitive<Time> {
  private zones: Zone[] = [];
  private firstTime = 0;
  private param: SeriesAttachedParameter<Time> | null = null;
  private readonly view: IPrimitivePaneView;

  constructor() {
    const renderer: IPrimitivePaneRenderer = {
      draw: () => {},
      drawBackground: (target: CanvasRenderingTarget2D) => this.drawZones(target),
    };
    this.view = { zOrder: () => "bottom", renderer: () => renderer };
  }

  attached(param: SeriesAttachedParameter<Time>) {
    this.param = param;
  }

  detached() {
    this.param = null;
  }

  paneViews() {
    return [this.view];
  }

  setZones(zones: Zone[], firstTime: number) {
    this.zones = zones;
    this.firstTime = firstTime;
    this.param?.requestUpdate();
  }

  private drawZones(target: CanvasRenderingTarget2D) {
    const p = this.param;
    if (!p) return;
    const ts = p.chart.timeScale();
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
      for (const z of this.zones) {
        const from = Math.max(z.from, this.firstTime);
        const x1: number | null = ts.timeToCoordinate(from as UTCTimestamp);
        if (x1 === null) continue;
        const x2raw = z.to === null ? null : ts.timeToCoordinate(z.to as UTCTimestamp);
        const x2 = x2raw === null ? mediaSize.width : x2raw;
        const yTop = p.series.priceToCoordinate(z.top);
        const yBot = p.series.priceToCoordinate(z.bottom);
        if (yTop === null || yBot === null) continue;
        const left = Math.min(x1, x2);
        const width = Math.max(2, Math.abs(x2 - x1));
        const top = Math.min(yTop, yBot);
        const height = Math.max(1, Math.abs(yBot - yTop));
        ctx.fillStyle = z.fill;
        ctx.fillRect(left, top, width, height);
        ctx.strokeStyle = z.stroke;
        ctx.lineWidth = 1;
        ctx.strokeRect(left + 0.5, top + 0.5, width - 1, height - 1);
        ctx.fillStyle = z.stroke;
        ctx.fillText(z.label, left + 3, top + 10);
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Chart component
// ---------------------------------------------------------------------------

const nyTime = new Intl.DateTimeFormat("en-US", {
  timeZone: TZ_NY,
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const nyDay = new Intl.DateTimeFormat("en-US", { timeZone: TZ_NY, month: "short", day: "numeric" });
const nyFull = new Intl.DateTimeFormat("en-US", {
  timeZone: TZ_NY,
  weekday: "short",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

function timeToDate(t: Time): Date {
  return new Date((typeof t === "number" ? t : 0) * 1000);
}

export interface CandleChartProps {
  tf: Timeframe;
  bars: Bar[];
  overlays: OverlaysResponse | null;
  signals: Signal[];
  showSwings: boolean;
}

const COLORS = {
  up: "#22c55e",
  down: "#ef4444",
  levelHigh: "#f472b6",
  levelLow: "#38bdf8",
  entry: "#60a5fa",
  stop: "#f87171",
  target: "#4ade80",
};

export default function CandleChart({ tf, bars, overlays, signals, showSwings }: CandleChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const zonesRef = useRef<ZonesPrimitive | null>(null);
  const linesRef = useRef<IPriceLine[]>([]);
  const dataRef = useRef<{ tf: Timeframe; first: number; len: number } | null>(null);

  // Create chart once.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const chart = createChart(el, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "#0f131c" },
        textColor: "#8b95a7",
        fontSize: 11,
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#222a38" },
      timeScale: {
        borderColor: "#222a38",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 8,
        tickMarkFormatter: (time: Time, type: TickMarkType) => {
          const d = timeToDate(time);
          if (type === TickMarkType.Time || type === TickMarkType.TimeWithSeconds) return nyTime.format(d);
          // day ticks fall on UTC midnight (20:00/19:00 NY the evening before): show the NY time too
          return type === TickMarkType.DayOfMonth ? `${nyDay.format(d)} ${nyTime.format(d)}` : nyDay.format(d);
        },
      },
      localization: {
        timeFormatter: (time: Time) => `${nyFull.format(timeToDate(time))} NY`,
        priceFormatter: (p: number) => p.toFixed(2),
      },
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderUpColor: COLORS.up,
      borderDownColor: COLORS.down,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
      priceLineVisible: true,
    });
    const zones = new ZonesPrimitive();
    series.attachPrimitive(zones);
    chartRef.current = chart;
    seriesRef.current = series;
    zonesRef.current = zones;
    markersRef.current = createSeriesMarkers(series, []);
    return () => {
      markersRef.current = null;
      zonesRef.current = null;
      seriesRef.current = null;
      chartRef.current = null;
      linesRef.current = [];
      dataRef.current = null;
      chart.remove();
    };
  }, []);

  // Push bar data (incremental update when only the last bar changed / one appended).
  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    const data: CandlestickData<Time>[] = bars.map((b) => ({
      time: isoToSec(b.t) as UTCTimestamp,
      open: b.o,
      high: b.h,
      low: b.l,
      close: b.c,
    }));
    const prev = dataRef.current;
    const first = data.length ? (data[0].time as number) : 0;
    const incremental =
      prev &&
      prev.tf === tf &&
      prev.first === first &&
      data.length >= prev.len &&
      data.length - prev.len <= 1 &&
      data.length > 0;
    if (incremental) {
      series.update(data[data.length - 1]);
    } else {
      series.setData(data);
      if (!prev || prev.tf !== tf) chartRef.current?.timeScale().fitContent();
    }
    dataRef.current = { tf, first, len: data.length };
  }, [bars, tf]);

  // Overlays, signal lines and markers.
  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    for (const l of linesRef.current) series.removePriceLine(l);
    linesRef.current = [];

    const firstTime = bars.length ? isoToSec(bars[0].t) : 0;
    const lastTime = bars.length ? isoToSec(bars[bars.length - 1].t) : 0;
    const snap = (iso: string) => bucketStart(isoToSec(iso), tf);
    const inRange = (sec: number) => bars.length > 0 && sec >= firstTime && sec <= lastTime;
    const markers: SeriesMarker<Time>[] = [];

    if (overlays) {
      for (const lv of overlays.levels) {
        const consumed = lv.consumed_at !== null;
        const color = lv.side === "high" ? COLORS.levelHigh : COLORS.levelLow;
        linesRef.current.push(
          series.createPriceLine({
            price: lv.price,
            color: consumed ? `${color}55` : color,
            lineWidth: 1,
            lineStyle: consumed ? LineStyle.SparseDotted : LineStyle.Dashed,
            axisLabelVisible: !consumed,
            title: consumed ? `${lv.kind} (taken)` : lv.kind,
          }),
        );
      }

      const zones: Zone[] = [];
      for (const f of overlays.fvgs) {
        const bull = f.direction === "bull";
        zones.push({
          from: snap(f.start),
          to: f.filled_at ? snap(f.filled_at) : null,
          top: f.top,
          bottom: f.bottom,
          fill: bull ? "rgba(34,197,94,0.10)" : "rgba(239,68,68,0.10)",
          stroke: bull ? "rgba(34,197,94,0.55)" : "rgba(239,68,68,0.55)",
          label: f.filled_at ? "FVG (filled)" : "FVG",
        });
      }
      for (const ob of overlays.order_blocks) {
        const bull = ob.direction === "bull";
        zones.push({
          from: snap(ob.start),
          to: null,
          top: ob.top,
          bottom: ob.bottom,
          fill: bull ? "rgba(56,189,248,0.08)" : "rgba(244,114,182,0.08)",
          stroke: bull ? "rgba(56,189,248,0.6)" : "rgba(244,114,182,0.6)",
          label: bull ? "Bull OB" : "Bear OB",
        });
      }
      zonesRef.current?.setZones(zones, firstTime);

      for (const s of overlays.sweeps) {
        const t = snap(s.t);
        if (!inRange(t)) continue;
        const short = s.direction === "short";
        markers.push({
          time: t as UTCTimestamp,
          position: short ? "aboveBar" : "belowBar",
          shape: short ? "arrowDown" : "arrowUp",
          color: "#fbbf24",
          text: `sweep ${s.level_kind}`,
        });
      }
      for (const m of overlays.mss) {
        const t = snap(m.t);
        if (!inRange(t)) continue;
        const short = m.direction === "short";
        markers.push({
          time: t as UTCTimestamp,
          position: short ? "belowBar" : "aboveBar",
          shape: "square",
          color: "#a78bfa",
          text: "MSS",
          size: 0.7,
        });
      }
      if (showSwings) {
        for (const s of overlays.swings) {
          const t = snap(s.t);
          if (!inRange(t)) continue;
          markers.push({
            time: t as UTCTimestamp,
            position: s.kind === "high" ? "aboveBar" : "belowBar",
            shape: "circle",
            color: "#64748b",
            size: 0.4,
          });
        }
      }
    } else {
      zonesRef.current?.setZones([], firstTime);
    }

    // Signals: markers for all in range; price lines for active ones (or the latest if none active).
    const active = signals.filter((s) => s.status === "active");
    const lined = active.length ? active : signals.slice(0, 1);
    for (const sig of signals) {
      const t = snap(sig.created_at);
      if (!inRange(t)) continue;
      const short = sig.direction === "short";
      markers.push({
        time: t as UTCTimestamp,
        position: short ? "aboveBar" : "belowBar",
        shape: short ? "arrowDown" : "arrowUp",
        color: short ? "#f87171" : "#4ade80",
        text: `#${sig.id} ${sig.direction.toUpperCase()}`,
        size: 1.4,
      });
    }
    for (const sig of lined) {
      const dim = sig.status !== "active";
      const style = dim ? LineStyle.Dotted : LineStyle.Solid;
      const mk = (price: number, color: string, title: string) =>
        series.createPriceLine({
          price,
          color: dim ? `${color}88` : color,
          lineWidth: dim ? 1 : 2,
          lineStyle: style,
          axisLabelVisible: true,
          title,
        });
      linesRef.current.push(
        mk(sig.entry, COLORS.entry, `#${sig.id} entry`),
        mk(sig.stop, COLORS.stop, `#${sig.id} SL`),
        mk(sig.target, COLORS.target, `#${sig.id} TP`),
      );
    }

    markers.sort((a, b) => (a.time as number) - (b.time as number));
    markersRef.current?.setMarkers(markers);
  }, [overlays, signals, bars, tf, showSwings]);

  return <div ref={containerRef} className="h-[340px] w-full sm:h-[440px] xl:h-[520px]" />;
}
