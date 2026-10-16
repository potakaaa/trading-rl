import type { Metadata, Viewport } from "next";
import { Suspense } from "react";
import { Nav } from "../components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "US100 Signal Assistant",
  description: "Paper-mode ICT signal dashboard (unproven signals)",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0b0e14",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col">
        <Suspense fallback={<div className="h-10 border-b border-edge" />}>
          <Nav />
        </Suspense>
        <main className="mx-auto w-full max-w-[1600px] flex-1 px-4 py-4">{children}</main>
      </body>
    </html>
  );
}
