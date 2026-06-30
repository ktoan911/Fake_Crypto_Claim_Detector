"use client";

import { useEffect, useRef, useState } from "react";
import { Activity, Terminal } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://flashcard-eldercare-throng.ngrok-free.dev";

type Status = "loading" | "connected" | "done" | "empty" | "error";
type Level = "INFO" | "WARN" | "ERROR" | "RAW";

interface ParsedLine {
  ts: string | null;
  level: Level;
  message: string;
}

// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;]*[mGKHF]/g;
function stripAnsi(s: string): string { return s.replace(ANSI_RE, ""); }

/** Strip \r, merge timestamp split: "21:40:41" + ",829 - INFO - msg" → "21:40:41,829 - INFO - msg" */
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

function toLevel(tag: string): Level {
  const t = tag.toUpperCase().trim();
  if (t === "ERROR") return "ERROR";
  if (t.startsWith("WARN")) return "WARN";
  if (t === "DEBUG") return "WARN";
  return "INFO";
}

function parseLine(raw: string): ParsedLine {
  const s = stripAnsi(raw);

  // Format: YYYY-MM-DD HH:MM:SS,mmm [LEVEL] message  (crawler format)
  const m1 = s.match(/^\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2})(?:[,.:]\d+)?\s+\[(INFO|WARNING|WARN|ERROR|DEBUG)\]\s*(.*)/i);
  if (m1) return { ts: m1[1], level: toLevel(m1[2]), message: m1[3] };

  // Format: loguru — YYYY-MM-DD HH:MM:SS.mmm | LEVEL | src:func:line - message
  const m2 = s.match(/^\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2})(?:[,.]\d+)?\s+\|\s*(INFO|WARNING|WARN|ERROR|DEBUG|SUCCESS)\s*\|[^-]+-\s*(.*)/i);
  if (m2) return { ts: m2[1], level: toLevel(m2[2]), message: m2[3] };

  // Format: HH:MM:SS,mmm - LEVEL - message
  const m3 = s.match(/^(\d{2}:\d{2}:\d{2}(?:,\d+)?)\s+-\s+(INFO|WARNING|WARN|ERROR|DEBUG)\s+-\s+(.*)/i);
  if (m3) return { ts: m3[1].replace(/,\d+$/, ""), level: toLevel(m3[2]), message: m3[3] };

  // Format: YYYY-MM-DD HH:MM:SS,mmm - LEVEL - message
  const m4 = s.match(/^\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2})(?:,\d+)?\s+-\s+(INFO|WARNING|WARN|ERROR|DEBUG)\s+-\s+(.*)/i);
  if (m4) return { ts: m4[1], level: toLevel(m4[2]), message: m4[3] };

  // TF/CUDA: E0000 00:00:... hoặc W0000 ...
  const m5 = s.match(/^([EW])\d{4}\s+\S+\s+\d+\s+(.*)/);
  if (m5) return { ts: null, level: m5[1] === "E" ? "ERROR" : "WARN", message: m5[2] };

  return { ts: null, level: "RAW", message: s };
}

const LEVEL_COLOR: Record<Level, string> = {
  INFO:  "text-emerald-400",
  WARN:  "text-yellow-300",
  ERROR: "text-red-400",
  RAW:   "text-slate-500",
};

const FILTER_LEVELS = ["ERROR", "WARN", "INFO"] as const;
type FilterLevel = typeof FILTER_LEVELS[number];

const FILTER_STYLE: Record<FilterLevel, { active: string; idle: string }> = {
  ERROR: { active: "bg-red-500/20 text-red-300 border-red-500/40",       idle: "text-slate-500 border-white/[0.06] hover:border-red-500/30 hover:text-red-400" },
  WARN:  { active: "bg-yellow-500/15 text-yellow-300 border-yellow-500/40", idle: "text-slate-500 border-white/[0.06] hover:border-yellow-500/30 hover:text-yellow-300" },
  INFO:  { active: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40", idle: "text-slate-500 border-white/[0.06] hover:border-emerald-500/30 hover:text-emerald-400" },
};

const MSG_COLOR: Record<Level, string> = {
  INFO:  "text-slate-200",
  WARN:  "text-yellow-100",
  ERROR: "text-red-200",
  RAW:   "text-white",
};

function LogRow({ line, index }: { line: string; index: number }) {
  const { ts, level, message } = parseLine(line);
  const zebra = index % 2 === 0 ? "bg-white/[0.015]" : "";

  return (
    <div
      title={message}
      className={`group flex items-center gap-0 py-1 px-4 hover:bg-white/[0.06] transition-colors cursor-default ${zebra}`}
    >
      <span className="shrink-0 w-[72px] mr-3 text-[11px] text-slate-600 tabular-nums">
        {ts ?? ""}
      </span>
      <span className={`shrink-0 w-11 mr-3 text-[11px] font-bold uppercase ${level !== "RAW" ? LEVEL_COLOR[level] : "text-transparent"}`}>
        {level !== "RAW" ? level : "—"}
      </span>
      <span className={`flex-1 min-w-0 text-[12.5px] truncate ${MSG_COLOR[level]}`}>
        {message}
      </span>
    </div>
  );
}

export default function CrawlingLog() {
  const [lines, setLines] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<Status>("loading");
  const [docId, setDocId] = useState<string | null>(null);
  const [filterLevel, setFilterLevel] = useState<FilterLevel | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevLenRef = useRef(0);

  useEffect(() => {
    async function poll() {
      try {
        const res = await fetch(`${API_URL}/crawler/logs`, {
          cache: "no-store",
          headers: { "ngrok-skip-browser-warning": "true" },
        });
        if (res.status === 404) { setStatus("empty"); return; }
        if (!res.ok) { setStatus((s) => s === "loading" ? "error" : s); return; }
        const data = await res.json();
        setLines(splitContent(data.content as string));
        setRunning(!!data.running);
        setDocId(data.doc_id ?? null);
        setStatus(data.running ? "connected" : "done");
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

  const displayed = filterLevel
    ? lines.filter((l) => parseLine(l).level === filterLevel)
    : lines;

  const statusDot =
    status === "connected" ? "bg-emerald-400 animate-pulse"
    : status === "done"      ? "bg-emerald-400"
    : status === "loading"   ? "bg-yellow-300 animate-pulse"
    : "bg-red-400";

  const statusLabel =
    status === "loading"   ? "loading…"
    : status === "connected" ? "running"
    : status === "done"      ? "completed"
    : status === "empty"     ? "no logs"
    : "error";

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
          <Terminal className="h-4 w-4 shrink-0 text-purple-400" />
          <span className="font-mono text-sm text-slate-400">
            crawler.log
            <span className="text-slate-600"> — shell stdout</span>
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="flex items-center gap-1 font-mono text-[11px]">
            <button
              onClick={() => setFilterLevel(null)}
              className={`rounded-full border px-2.5 py-0.5 transition-colors ${
                filterLevel === null
                  ? "bg-white/10 text-slate-200 border-white/20"
                  : "text-slate-500 border-white/[0.06] hover:text-slate-300 hover:border-white/20"
              }`}
            >
              All
            </button>
            {FILTER_LEVELS.map((lvl) => (
              <button
                key={lvl}
                onClick={() => setFilterLevel(filterLevel === lvl ? null : lvl)}
                className={`rounded-full border px-2.5 py-0.5 transition-colors ${
                  filterLevel === lvl ? FILTER_STYLE[lvl].active : FILTER_STYLE[lvl].idle
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>
          <span className="hidden h-4 w-px bg-white/10 sm:block" />
          <div className="hidden items-center gap-1.5 rounded-full border border-purple-500/20 bg-purple-500/10 px-3 py-1 font-mono text-xs text-purple-400 sm:flex">
            <Activity className="h-3 w-3" />
            tail -f
          </div>
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center gap-0 border-b border-white/[0.04] px-4 py-1.5">
        <span className="w-20 mr-3 text-[10px] uppercase tracking-widest text-slate-600">Time</span>
        <span className="w-12 mr-3 text-[10px] uppercase tracking-widest text-slate-600">Level</span>
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
        {displayed.length === 0 && (status === "connected" || status === "done") && (
          <p className="px-4 py-3 text-sm text-slate-500">
            {filterLevel ? `Không có dòng ${filterLevel} nào.` : "Không có log nào."}
          </p>
        )}
        {displayed.map((line, i) => <LogRow key={i} line={line} index={i} />)}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between border-t border-white/[0.06] px-5 py-2.5 font-mono text-xs text-slate-600">
        <div className="flex items-center gap-2">
          <span className={`h-1.5 w-1.5 rounded-full ${statusDot}`} />
          <span>crawler-shell · {statusLabel}</span>
        </div>
        <div className="hidden gap-4 sm:flex">
          {docId && <span className="text-slate-700" title="doc_id đang hiển thị">doc: {docId}</span>}
          <span>{filterLevel ? `${displayed.length} / ${lines.length} lines` : `${lines.length} lines`}</span>
          {running && <span className="text-purple-400">● live</span>}
        </div>
      </div>
    </div>
  );
}
