"use client";

import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import {
  BarChart3,
  ChevronDown,
  FileText,
  LayoutDashboard,
  LogOut,
  Server,
  User,
} from "lucide-react";

import { useEffect } from "react";
import { ThemeToggleButton } from "@/src/components/ui/ThemeToggleButton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/src/components/ui/dropdown-menu";

type AdminShellProps = {
  children: React.ReactNode;
};

type AdminMenuItem = {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
};

const ADMIN_MENU_ITEMS: AdminMenuItem[] = [
  { label: "Tổng quan",      href: "/admin",                  icon: LayoutDashboard },
  { label: "Phân Tích",     href: "/admin#campaigns",        icon: BarChart3       },
  { label: "Crawling Logs", href: "/admin#crawling-logs",   icon: FileText        },
  { label: "Server Logs",   href: "/admin#server-logs",     icon: Server          },
];

const fakeUser = { name: "Admin User" };

const GRADIENT = "linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)";

export default function HeaderAdmin({ children }: AdminShellProps) {
  const [selectedHref, setSelectedHref] = useState("/admin");

  useEffect(() => {
    const sectionMap: Record<string, string> = {
      "content-report": "/admin",
      campaigns: "/admin#campaigns",
      "crawling-logs": "/admin#crawling-logs",
      "server-logs": "/admin#server-logs",
    };

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          const href = sectionMap[visible[0].target.id];
          if (href) setSelectedHref(href);
        }
      },
      { threshold: 0.3 }
    );

    Object.keys(sectionMap).forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen bg-[var(--body-bg)] text-foreground">

      {/* ── Sidebar ────────────────────────────────────────────────────── */}
      <aside
        className="fixed inset-y-0 left-0 z-50 hidden md:flex flex-col w-64 bg-white dark:bg-[#111217]"
        style={{
          borderRight: "1px solid rgba(0,0,0,0.06)",
          boxShadow: "2px 0 16px rgba(0,0,0,0.04)",
        }}
      >
        {/* Logo */}
        <div className="p-5 pb-4 border-b border-gray-100 dark:border-slate-800">
          <div className="flex items-center gap-2.5">
            <Image src="/LOGO.png" alt="Logo" width={32} height={32} className="shrink-0" style={{ objectFit: "contain" }} />
            <div className="min-w-0">
              <h1
                className="text-sm font-semibold bg-clip-text text-transparent"
                style={{ backgroundImage: GRADIENT }}
              >
                System Monitor
              </h1>
              <p className="text-xs text-gray-400 dark:text-slate-500">
                Monitoring Dashboard
              </p>
            </div>
          </div>
        </div>

        {/* Section label */}
        <div className="px-5 pt-5 pb-2">
          <span className="text-xs font-medium tracking-widest text-gray-400 dark:text-slate-500">
            Điều hướng
          </span>
        </div>

        {/* Nav items */}
        <nav className="flex-1 px-3 space-y-0.5 overflow-y-auto">
          {ADMIN_MENU_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = selectedHref === item.href;

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setSelectedHref(item.href)}
                className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 group relative ${
                  isActive
                    ? "text-white shadow-md shadow-purple-500/25"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
                }`}
                style={isActive ? { background: GRADIENT } : {}}
              >
                {/* Glow behind active item */}
                {isActive && (
                  <div
                    className="absolute inset-0 rounded-xl opacity-30 blur-sm"
                    style={{ background: GRADIENT }}
                  />
                )}

                <Icon
                  className={`w-4 h-4 relative z-10 shrink-0 ${
                    isActive
                      ? "text-white"
                      : "text-gray-400 group-hover:text-gray-600 dark:text-slate-500 dark:group-hover:text-slate-300"
                  }`}
                />

                <span className="text-sm relative z-10">{item.label}</span>

                {isActive && (
                  <div className="ml-auto relative z-10 w-1.5 h-1.5 rounded-full bg-white/60" />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="pb-4" />
      </aside>

      {/* ── Top header ─────────────────────────────────────────────────── */}
      <header
        className="fixed right-0 top-0 z-40 h-14 left-0 md:left-64 flex items-center justify-between px-4 md:px-6 bg-white/90 dark:bg-background/95 backdrop-blur"
        style={{
          borderBottom: "1px solid rgba(0,0,0,0.06)",
          boxShadow: "0 1px 8px rgba(0,0,0,0.04)",
        }}
      >
        {/* Mobile: logo */}
        <div className="flex items-center gap-2 md:hidden">
          <Image src="/LOGO.png" alt="Logo" width={28} height={28} style={{ objectFit: "contain" }} />
          <span
            className="text-sm font-bold bg-clip-text text-transparent"
            style={{ backgroundImage: GRADIENT }}
          >
            System Monitor
          </span>
        </div>

        {/* Right actions */}
        <div className="flex items-center gap-2 ml-auto">
          <ThemeToggleButton />

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                className="flex items-center gap-2 px-2 py-1.5 rounded-xl hover:bg-gray-50 dark:hover:bg-slate-800 transition-colors"
              >
                <div
                  className="w-7 h-7 rounded-full flex items-center justify-center shadow shadow-purple-400/30 shrink-0"
                  style={{ background: GRADIENT }}
                >
                  <User className="w-4 h-4 text-white" />
                </div>
                <div className="hidden sm:block text-left">
                  <p className="text-xs font-medium text-gray-700 dark:text-slate-200">
                    {fakeUser.name}
                  </p>
                </div>
                <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem className="gap-2">
                <User className="h-4 w-4" />
                {fakeUser.name}
              </DropdownMenuItem>
              <DropdownMenuItem
                className="gap-2 text-red-500 focus:text-red-500 cursor-pointer"
                onClick={() => {
                  document.cookie = "admin_auth=; path=/; max-age=0";
                  window.location.href = "/admin/login";
                }}
              >
                <LogOut className="h-4 w-4" />
                Đăng xuất
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────────────── */}
      <main className="min-h-screen pt-14 md:pl-64">
        <div className="min-w-0 px-4 py-6 md:px-8">{children}</div>
      </main>
    </div>
  );
}
