import type { Metadata } from "next";
import { JournalView } from "../../components/JournalView";

export const metadata: Metadata = { title: "Journal · US100 Signal Assistant" };

export default function JournalPage() {
  return <JournalView />;
}
