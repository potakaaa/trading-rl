"use client";

import { api, errorText, isOffline } from "../lib/api";
import { useApi } from "../lib/hooks";
import { BackendOffline, ErrorNote, JsonView, Loading } from "./ui";

export function SettingsView() {
  const settings = useApi("settings", (s) => api.settings(s));
  if (!settings.data) {
    if (settings.error)
      return isOffline(settings.error) ? <BackendOffline /> : <ErrorNote text={errorText(settings.error)} />;
    return <Loading />;
  }
  const entries = Object.entries(settings.data);
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Settings</h1>
        <span className="text-xs text-muted">read-only · settings.yaml (secrets excluded)</span>
      </div>
      {entries.map(([k, v]) => (
        <details key={k} className="rounded-lg border border-edge bg-panel" open={entries.length <= 3}>
          <summary className="cursor-pointer select-none px-3 py-2 font-mono text-sm font-semibold">{k}</summary>
          <div className="border-t border-edge p-3">
            <JsonView value={v} depth={1} />
          </div>
        </details>
      ))}
    </div>
  );
}
