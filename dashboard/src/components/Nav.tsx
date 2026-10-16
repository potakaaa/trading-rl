"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Live" },
  { href: "/backtest", label: "Backtest" },
  { href: "/journal", label: "Journal" },
  { href: "/settings", label: "Settings" },
];

export function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b border-edge bg-bg/90 backdrop-blur">
      <div className="mx-auto flex max-w-[1600px] items-center gap-4 px-4 py-2">
        <Link href="/" className="shrink-0 text-sm font-bold tracking-wide">
          US100 <span className="hidden text-muted sm:inline">signal assistant</span>
        </Link>
        <nav className="flex gap-1 overflow-x-auto">
          {LINKS.map((l) => {
            const on = l.href === "/" ? path === "/" : path?.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded px-2.5 py-1 text-sm ${
                  on ? "bg-white/10 text-fg" : "text-muted hover:text-fg"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
