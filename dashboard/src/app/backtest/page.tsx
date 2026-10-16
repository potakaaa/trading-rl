import type { Metadata } from "next";
import { BacktestView } from "../../components/BacktestView";

export const metadata: Metadata = { title: "Backtest · US100 Signal Assistant" };

export default function BacktestPage() {
  return <BacktestView />;
}
