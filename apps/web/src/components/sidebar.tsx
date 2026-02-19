"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

interface NavItem {
  href: string;
  label: string;
  icon: React.ReactNode;
  match?: (path: string) => boolean;
}

const nav: NavItem[] = [
  {
    href: "/",
    label: "Dashboard",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="1.5" y="1.5" width="5" height="5" rx="1" />
        <rect x="9.5" y="1.5" width="5" height="5" rx="1" />
        <rect x="1.5" y="9.5" width="5" height="5" rx="1" />
        <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
      </svg>
    ),
    match: (p) => p === "/",
  },
  {
    href: "/training",
    label: "Live Training",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="8" cy="8" r="3" />
        <path d="M4.5 4.5L2 2M11.5 4.5L14 2M4.5 11.5L2 14M11.5 11.5L14 14" />
      </svg>
    ),
    match: (p) => p === "/training",
  },
  {
    href: "/runs",
    label: "Training Runs",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="1.5,12 5,6 8.5,9 14.5,3" />
        <polyline points="10.5,3 14.5,3 14.5,7" />
      </svg>
    ),
    match: (p) => p === "/runs" || p.startsWith("/runs/"),
  },
  {
    href: "/domains",
    label: "Domains",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="8" cy="8" r="6.5" />
        <ellipse cx="8" cy="8" rx="3" ry="6.5" />
        <line x1="1.5" y1="8" x2="14.5" y2="8" />
      </svg>
    ),
    match: (p) => p.startsWith("/domains"),
  },
  {
    href: "/checkpoints",
    label: "Checkpoints",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 10v3a1 1 0 01-1 1H3a1 1 0 01-1-1v-3" />
        <polyline points="5,6 8,2 11,6" />
        <line x1="8" y1="2" x2="8" y2="11" />
      </svg>
    ),
    match: (p) => p.startsWith("/checkpoints"),
  },
  {
    href: "/models",
    label: "Models",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M8 1.5L14 5v6l-6 3.5L2 11V5l6-3.5z" />
        <path d="M8 8.5V15" />
        <path d="M2 5l6 3.5L14 5" />
      </svg>
    ),
    match: (p) => p === "/models" || p.startsWith("/models/"),
  },
];

const tools: NavItem[] = [
  {
    href: "/inference",
    label: "Inference",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="5,2 13,8 5,14" />
      </svg>
    ),
    match: (p) => p === "/inference",
  },
  {
    href: "/chat",
    label: "Chat",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 3h12a1 1 0 011 1v7a1 1 0 01-1 1H5l-3 3V4a1 1 0 011-1z" />
      </svg>
    ),
    match: (p) => p === "/chat",
  },
  {
    href: "/docs",
    label: "API Docs",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 1.5h7l4 4v9a1 1 0 01-1 1H3a1 1 0 01-1-1v-12a1 1 0 011-1z" />
        <polyline points="10,1.5 10,5.5 14,5.5" />
      </svg>
    ),
    match: (p) => p === "/docs",
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 z-20 flex h-full w-56 flex-col border-r border-border bg-surface">
      {/* Brand */}
      <div className="flex h-14 items-center gap-2.5 px-5">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent text-xs font-bold text-white">
          A
        </div>
        <span className="text-sm font-semibold text-white">Alpha</span>
      </div>

      {/* Main nav */}
      <nav className="flex-1 space-y-0.5 px-3 pt-2">
        <div className="mb-2 px-2 text-[0.6rem] font-semibold uppercase tracking-widest text-text-muted">
          Training
        </div>
        {nav.map((item) => {
          const active = item.match
            ? item.match(pathname)
            : pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[0.8rem] transition-colors hover:no-underline ${
                active
                  ? "bg-surface-2 font-medium text-white"
                  : "text-text-secondary hover:bg-surface-2 hover:text-text-primary"
              }`}
            >
              <span className={active ? "text-accent" : "text-text-muted"}>
                {item.icon}
              </span>
              {item.label}
            </Link>
          );
        })}

        <div className="mb-2 mt-6 px-2 text-[0.6rem] font-semibold uppercase tracking-widest text-text-muted">
          Tools
        </div>
        {tools.map((item) => {
          const active = item.match
            ? item.match(pathname)
            : pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[0.8rem] transition-colors hover:no-underline ${
                active
                  ? "bg-surface-2 font-medium text-white"
                  : "text-text-secondary hover:bg-surface-2 hover:text-text-primary"
              }`}
            >
              <span className={active ? "text-accent" : "text-text-muted"}>
                {item.icon}
              </span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-border px-5 py-3">
        <div className="text-[0.6rem] text-text-muted">
          alpha.omegaai.dev
        </div>
      </div>
    </aside>
  );
}

export function MobileHeader() {
  const pathname = usePathname();
  const current = nav.find((item) =>
    item.match ? item.match(pathname) : pathname === item.href
  );

  return (
    <header className="sticky top-0 z-10 flex h-14 items-center gap-3 border-b border-border bg-surface px-4 lg:hidden">
      <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent text-xs font-bold text-white">
        A
      </div>
      <span className="text-sm font-semibold text-white">Alpha</span>
      <nav className="ml-4 flex gap-1">
        {nav.map((item) => {
          const active = item.match
            ? item.match(pathname)
            : pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`rounded-md px-2.5 py-1.5 text-xs transition-colors hover:no-underline ${
                active
                  ? "bg-surface-2 font-medium text-white"
                  : "text-text-secondary hover:text-text-primary"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
