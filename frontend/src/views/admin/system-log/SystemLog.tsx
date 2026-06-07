"use client";

import { useEffect, useRef, useState } from "react";
import { Activity, Server } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

type Status = "loading" | "connected" | "done" | "empty" | "error";
type Level = "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL" | "RAW";

interface ParsedLine {
  ts: string | null;
  level: Level;
  module: string | null;
  message: string;
}

/** Strip \r, merge lines split by \r\n */
function splitContent(content: string): string[] {
  const raw = content.split("\n").map((l) => l.replace(/\r/g, ""));
  const merged: string[] = [];
  for (const line of raw) {
    if (line.startsWith(",") && merged.length > 0) {
      merged[merged.length - 1] += line;
    } else if (line.trim()) {
      merged.push(line);
    }
  }
  return merged;
}

function parseLine(raw: string): ParsedLine {
  // Loguru full format: 2026-06-06 21:27:43.996 | WARNING  | module:func:line - message
  const m = raw.match(
    /^\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2})\.\d+\s+\|\s+(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SUCCESS)\s+\|\s+([^-]+?)\s+-\s+(.*)/i
  );
  if (m) {
    const level = normalizeLevel(m[2]);
    return { ts: m[1], level, module: m[3].trim(), message: m[4] };
  }

  // Our custom sink format: HH:MM:SS - LEVEL - message
  const m2 = raw.match(
    /^(\d{2}:\d{2}:\d{2}(?:,\d+)?)\s+-\s+(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SUCCESS)\s+-\s+(.*)/i
  );
  if (m2) {
    return { ts: m2[1].replace(/,\d+$/, ""), level: normalizeLevel(m2[2]), module: null, message: m2[3] };
  }

  // Uvicorn format: INFO:     message
  const m3 = raw.match(/^(DEBUG|INFO|WARNING|ERROR|CRITICAL):\s+(.*)/i);
  if (m3) {
    return { ts: null, level: normalizeLevel(m3[1]), module: null, message: m3[2] };
  }

  return { ts: null, level: "RAW", module: null, message: raw };
}

function normalizeLevel(s: string): Level {
  const u = s.toUpperCase();
  if (u === "WARNING" || u === "WARN") return "WARNING";
  if (u === "CRITICAL") return "CRITICAL";
  if (u === "ERROR") return "ERROR";
  if (u === "DEBUG") return "DEBUG";
  if (u === "INFO" || u === "SUCCESS") return "INFO";
  return "RAW";
}

const LEVEL_COLOR: Record<Level, string> = {
  DEBUG:    "text-slate-500",
  INFO:     "text-cyan-400",
  WARNING:  "text-yellow-300",
  ERROR:    "text-red-400",
  CRITICAL: "text-red-500",
  RAW:      "text-slate-500",
};

const MSG_COLOR: Record<Level, string> = {
  DEBUG:    "text-slate-500",
  INFO:     "text-slate-100",
  WARNING:  "text-yellow-100",
  ERROR:    "text-red-200",
  CRITICAL: "text-red-200",
  RAW:      "text-slate-400",
};

const MODULE_SHORTENED: Record<string, never> = {} as never;

function shorten(module: string | null): string | null {
  if (!module) return null;
  // src.models.fusion_inference:__init__:453 → fusion_inference:453
  const parts = module.split(".");
  const last = parts[parts.length - 1]; // "fusion_inference:__init__:453"
  const segs = last.split(":");
  if (segs.length >= 3) return `${segs[0]}:${segs[segs.length - 1]}`; // "fusion_inference:453"
  return last;
}

function LogRow({ line, index }: { line: string; index: number }) {
  const { ts, level, module, message } = parseLine(line);
  const zebra = index % 2 === 0 ? "bg-white/[0.015]" : "";
  const short = shorten(module);

  return (
    <div
      title={`${module ? `[${module}] ` : ""}${message}`}
      className={`group flex items-center gap-0 py-[3px] px-4 hover:bg-white/[0.06] transition-colors cursor-default ${zebra}`}
    >
      {/* Time */}
      <span className="shrink-0 w-[72px] mr-3 text-[11px] text-slate-600 tabular-nums">
        {ts ?? ""}
      </span>
      {/* Level */}
      <span className={`shrink-0 w-16 mr-2 text-[11px] font-bold uppercase ${level !== "RAW" ? LEVEL_COLOR[level] : "text-transparent"}`}>
        {level !== "RAW" ? level : ""}
      </span>
      {/* Module (shortened) */}
      {short && (
        <span className="shrink-0 mr-2 text-[11px] text-slate-600 tabular-nums hidden lg:inline">
          {short}
        </span>
      )}
      {/* Message */}
      <span className={`flex-1 min-w-0 text-[12.5px] truncate ${MSG_COLOR[level]}`}>
        {message}
      </span>
    </div>
  );
}

export default function SystemLog() {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevLenRef = useRef(0);

  useEffect(() => {
    async function poll() {
      try {
        const res = await fetch(`${API_URL}/server/logs`, { cache: "no-store" });
        if (res.status === 404) { setStatus("empty"); return; }
        if (!res.ok) { setStatus((s) => s === "loading" ? "error" : s); return; }
        const data = await res.json();
        const next = splitContent(data.content as string);
        setLines(next);
        setStatus("connected");
      } catch {
        setStatus((s) => s === "loading" ? "error" : s);
      }
    }

    poll();
    const timer = window.setInterval(poll, 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (lines.length > prevLenRef.current) {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
    prevLenRef.current = lines.length;
  }, [lines]);

  const statusDot =
    status === "connected" ? "bg-emerald-400 animate-pulse"
    : status === "loading"   ? "bg-yellow-300 animate-pulse"
    : status === "done"      ? "bg-emerald-400"
    : "bg-red-400";

  return (
    <div
      className="mt-8 overflow-hidden rounded-2xl shadow-lg bg-[#0d1117]"
      style={{ border: "1px solid rgba(139,92,246,0.15)" }}
    >
      {/* Header */}
      <div className="flex h-14 items-center justify-between gap-4 border-b border-white/[0.06] px-5">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex gap-1.5">
            <span className="h-3 w-3 rounded-full bg-red-500/80" />
            <span className="h-3 w-3 rounded-full bg-yellow-400/80" />
            <span className="h-3 w-3 rounded-full bg-emerald-500/80" />
          </div>
          <span className="h-5 w-px bg-white/10" />
          <Server className="h-4 w-4 shrink-0 text-cyan-400" />
          <span className="font-mono text-sm text-slate-400">
            server.log
            <span className="text-slate-600"> — api-server stdout</span>
          </span>
        </div>
        <div className="hidden items-center gap-1.5 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 font-mono text-xs text-cyan-400 sm:flex">
          <Activity className="h-3 w-3" />
          tail -f
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center border-b border-white/[0.04] px-4 py-1.5">
        <span className="w-[72px] mr-3 text-[10px] uppercase tracking-widest text-slate-600">Time</span>
        <span className="w-16 mr-2 text-[10px] uppercase tracking-widest text-slate-600">Level</span>
        <span className="mr-2 text-[10px] uppercase tracking-widest text-slate-600 hidden lg:inline w-32">Module</span>
        <span className="text-[10px] uppercase tracking-widest text-slate-600">Message</span>
      </div>

      {/* Log body */}
      <div
        ref={scrollRef}
        className="h-[500px] overflow-y-auto overflow-x-hidden py-1 font-mono"
      >
        {status === "loading" && <p className="px-4 py-3 text-sm text-slate-500">Đang tải log…</p>}
        {status === "empty"   && <p className="px-4 py-3 text-sm text-slate-500">Chưa có log nào.</p>}
        {status === "error"   && <p className="px-4 py-3 text-sm text-red-400">Không thể tải log từ API.</p>}
        {lines.map((line, i) => <LogRow key={i} line={line} index={i} />)}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between border-t border-white/[0.06] px-5 py-2.5 font-mono text-xs text-slate-600">
        <div className="flex items-center gap-2">
          <span className={`h-1.5 w-1.5 rounded-full ${statusDot}`} />
          <span>api-server · {status === "connected" ? "live" : status}</span>
        </div>
        <span>{lines.length} lines</span>
      </div>
    </div>
  );
}
