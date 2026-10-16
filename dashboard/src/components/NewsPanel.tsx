"use client";

import { useState } from "react";
import { errorText, type NewsResponse, type NewsStatus } from "../lib/api";
import { TZ_MANILA, TZ_NY, fmtCountdown, fmtDateTime, fmtDay, fmtTime, ymd } from "../lib/format";
import { ErrorNote, Loading, Panel } from "./ui";

export function NewsPanel({
  news,
  error,
  live,
  now,
}: {
  news: NewsResponse | undefined;
  error: unknown;
  live: NewsStatus | undefined;
  now: number | null;
}) {
  const [range, setRange] = useState<"today" | "week">("today");
  const todayNy = now === null ? null : ymd(now, TZ_NY);
  const events = (news?.events ?? []).slice().sort((a, b) => a.ts.localeCompare(b.ts));
  const shown = range === "today" && todayNy ? events.filter((e) => ymd(e.ts, TZ_NY) === todayNy) : events;

  // Countdown target: status.next_event if present, else first future event from the list.
  const nextFromList = now === null ? undefined : events.find((e) => new Date(e.ts).getTime() > now);
  const next = live?.next_event ?? (nextFromList ? { title: nextFromList.title, ts: nextFromList.ts } : null);
  const countdown = next && now !== null ? new Date(next.ts).getTime() - now : null;
  const available = live?.available ?? news?.available ?? true;

  return (
    <Panel
      title="News (USD high impact)"
      right={
        <div className="flex gap-1">
          <button type="button" className={`btn ${range === "today" ? "btn-on" : ""}`} onClick={() => setRange("today")}>
            Today
          </button>
          <button type="button" className={`btn ${range === "week" ? "btn-on" : ""}`} onClick={() => setRange("week")}>
            Week
          </button>
        </div>
      }
    >
      <div className="mb-3 grid gap-2 sm:grid-cols-2">
        <div
          className={`rounded border px-2 py-1.5 text-xs ${
            live?.blackout ? "border-amber-500/50 bg-amber-500/15 text-amber-200" : "border-edge text-slate-300"
          }`}
        >
          <div className="text-[10px] uppercase tracking-wider text-muted">Blackout</div>
          <div className="font-semibold">{live ? (live.blackout ? "IN BLACKOUT" : "clear") : "–"}</div>
          {live?.reason ? <div className="text-[11px]">{live.reason}</div> : null}
        </div>
        <div className="rounded border border-edge px-2 py-1.5 text-xs">
          <div className="text-[10px] uppercase tracking-wider text-muted">Next event</div>
          {next ? (
            <>
              <div className="truncate font-semibold">{next.title}</div>
              <div className="tabular-nums text-sky-300">
                {countdown === null ? "–" : countdown > 0 ? `in ${fmtCountdown(countdown)}` : "now / passed"}
              </div>
              <div className="text-[11px] text-muted">
                NY {fmtDateTime(next.ts, TZ_NY)} · MNL {fmtDateTime(next.ts, TZ_MANILA)}
              </div>
            </>
          ) : (
            <div className="text-muted">none this week</div>
          )}
        </div>
      </div>

      {!available ? (
        <div className="mb-2">
          <ErrorNote text="News feed unavailable – engine treats this as NO-NEWS (no signals)." />
        </div>
      ) : null}
      {error && !news ? <ErrorNote text={errorText(error)} /> : null}
      {!news && !error ? <Loading /> : null}

      {news ? (
        shown.length === 0 ? (
          <p className="text-sm text-muted">No events {range === "today" ? "today" : "this week"}.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[440px] text-xs">
              <thead>
                <tr className="text-left text-[10px] uppercase tracking-wider text-muted">
                  <th className="py-1 pr-2 font-medium">Event</th>
                  <th className="py-1 pr-2 font-medium">New York</th>
                  <th className="py-1 pr-2 font-medium">Manila</th>
                  <th className="py-1 font-medium">Blackout (NY)</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((e, i) => {
                  const t = new Date(e.ts).getTime();
                  const inWindow =
                    now !== null && now >= new Date(e.blackout_start).getTime() && now <= new Date(e.blackout_end).getTime();
                  const past = now !== null && t < now && !inWindow;
                  return (
                    <tr
                      key={`${e.ts}-${i}`}
                      className={`border-t border-edge/60 ${inWindow ? "bg-amber-500/10" : ""} ${past ? "text-slate-500" : ""}`}
                    >
                      <td className="py-1 pr-2">{e.title}</td>
                      <td className="py-1 pr-2 tabular-nums">
                        {range === "week" ? `${fmtDay(e.ts, TZ_NY)} ` : ""}
                        {fmtTime(e.ts, TZ_NY)}
                      </td>
                      <td className="py-1 pr-2 tabular-nums">
                        {range === "week" ? `${fmtDay(e.ts, TZ_MANILA)} ` : ""}
                        {fmtTime(e.ts, TZ_MANILA)}
                      </td>
                      <td className="py-1 tabular-nums">
                        {fmtTime(e.blackout_start, TZ_NY)}–{fmtTime(e.blackout_end, TZ_NY)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )
      ) : null}
      {news?.last_refresh ? (
        <p className="mt-2 text-[10px] text-muted">Calendar refreshed NY {fmtDateTime(news.last_refresh, TZ_NY)}</p>
      ) : null}
    </Panel>
  );
}
