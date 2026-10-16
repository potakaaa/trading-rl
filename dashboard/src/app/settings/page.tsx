import type { Metadata } from "next";
import { SettingsView } from "../../components/SettingsView";

export const metadata: Metadata = { title: "Settings · US100 Signal Assistant" };

export default function SettingsPage() {
  return <SettingsView />;
}
