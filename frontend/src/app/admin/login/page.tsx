"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Lock,
  User,
  ArrowRight,
  AlertCircle,
  CheckCircle2,
  Eye,
  EyeOff,
  TrendingUp,
  Database,
  Activity,
  Cpu,
} from "lucide-react";
import Image from "next/image";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

// ── Types ─────────────────────────────────────────────────────────────────────
type AuthState = "idle" | "loading" | "error" | "success";

// ── Pipeline node graph ───────────────────────────────────────────────────────
const NODES = [
  { id: "mxh",   cx: 108, cy: 78,  label: "Mạng XH",  color: "#60a5fa", isHub: false, isKB: false, delay: 0    },
  { id: "bao",   cx: 92,  cy: 205, label: "Báo chí",  color: "#818cf8", isHub: false, isKB: false, delay: 0.15 },
  { id: "forum", cx: 108, cy: 332, label: "Diễn đàn", color: "#60a5fa", isHub: false, isKB: false, delay: 0.3  },
  { id: "hub",   cx: 288, cy: 205, label: "Crawler",  color: "#a78bfa", isHub: true,  isKB: false, delay: 0.45 },
  { id: "loc",   cx: 432, cy: 100, label: "Lọc",      color: "#818cf8", isHub: false, isKB: false, delay: 0.6  },
  { id: "kho",   cx: 520, cy: 218, label: "Kho KB",   color: "#34d399", isHub: false, isKB: true,  delay: 0.75 },
  { id: "ai",    cx: 432, cy: 330, label: "AI Model", color: "#818cf8", isHub: false, isKB: false, delay: 0.9  },
];

// Bezier path strings — "M sx sy Q qx qy ex ey"  or  "M sx sy L ex ey"
const EDGES = [
  { d: "M 108 78 Q 195 95 288 205",   packetColor: "#818cf8", delay: 0.2,  dur: 2.8 },
  { d: "M 92 205 Q 190 170 288 205",  packetColor: "#818cf8", delay: 0.35, dur: 2.2 },
  { d: "M 108 332 Q 195 310 288 205",  packetColor: "#818cf8", delay: 0.5,  dur: 2.8 },
  { d: "M 288 205 Q 358 140 432 100",  packetColor: "#60a5fa", delay: 0.65, dur: 2.5 },
  { d: "M 432 100 Q 488 155 520 218",  packetColor: "#34d399", delay: 0.8,  dur: 2.0 },
  { d: "M 520 218 Q 490 275 432 330",  packetColor: "#34d399", delay: 1.1,  dur: 2.2 },
];

const SCATTER = [
  { cx: 195, cy: 150, r: 2.5, delay: 0.5  },
  { cx: 355, cy: 272, r: 2,   delay: 0.9  },
  { cx: 162, cy: 290, r: 2,   delay: 1.2  },
  { cx: 368, cy: 158, r: 2.5, delay: 0.7  },
  { cx: 480, cy: 285, r: 2,   delay: 1.4  },
  { cx: 52,  cy: 270, r: 1.5, delay: 0.8  },
  { cx: 450, cy: 52,  r: 2,   delay: 0.4  },
  { cx: 222, cy: 318, r: 2,   delay: 1.6  },
  { cx: 308, cy: 68,  r: 2.5, delay: 0.6  },
  { cx: 558, cy: 150, r: 2,   delay: 1.0  },
  { cx: 148, cy: 130, r: 1.5, delay: 1.8  },
  { cx: 400, cy: 370, r: 2,   delay: 1.3  },
];

const PARTICLES = [
  { cx: 60,  dur: 6.5, delay: 0,   r: 1.8 },
  { cx: 192, dur: 8,   delay: 1.3, r: 1.5 },
  { cx: 332, dur: 7,   delay: 0.6, r: 2   },
  { cx: 452, dur: 9,   delay: 2.1, r: 1.5 },
  { cx: 542, dur: 6,   delay: 0.4, r: 2   },
  { cx: 152, dur: 10,  delay: 2.9, r: 1   },
  { cx: 415, dur: 7.5, delay: 1.7, r: 1.8 },
  { cx: 265, dur: 8.5, delay: 3.2, r: 1.5 },
];

function bezierPt(sx: number, sy: number, qx: number, qy: number, ex: number, ey: number, t: number) {
  const mt = 1 - t;
  return { x: mt * mt * sx + 2 * mt * t * qx + t * t * ex,
           y: mt * mt * sy + 2 * mt * t * qy + t * t * ey };
}

function DataPacket({ d, color, delay, dur }: { d: string; color: string; delay: number; dur: number }) {
  const isLine = d.includes(" L ");
  const pts: { x: number; y: number }[] = [];
  if (isLine) {
    const m = d.match(/M ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+)/);
    if (m) {
      const [, sx, sy, ex, ey] = m.map(Number);
      for (let i = 0; i <= 20; i++) pts.push({ x: sx + (ex - sx) * (i / 20), y: sy + (ey - sy) * (i / 20) });
    }
  } else {
    const m = d.match(/M ([\d.]+) ([\d.]+) Q ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)/);
    if (m) {
      const [, sx, sy, qx, qy, ex, ey] = m.map(Number);
      for (let i = 0; i <= 20; i++) pts.push(bezierPt(sx, sy, qx, qy, ex, ey, i / 20));
    }
  }
  if (!pts.length) return null;
  return (
    <motion.circle
      r={3}
      fill={color}
      filter="url(#pkgGlow)"
      animate={{ cx: pts.map((p) => p.x), cy: pts.map((p) => p.y), opacity: [0, 1, 1, 0] }}
      transition={{ duration: dur, delay, repeat: Infinity, ease: "linear", repeatDelay: 0.5 }}
    />
  );
}

function PipelineNode({ cx, cy, label, color, isHub, isKB, delay }: typeof NODES[0]) {
  const outer = isKB ? 31 : isHub ? 29 : 25;
  const body  = isKB ? 20 : isHub ? 18 : 15;
  const dot   = isKB ? 9  : isHub ? 7  : 6;
  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.6 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.55, delay, type: "spring", stiffness: 180, damping: 18 }}
      style={{ originX: `${cx}px`, originY: `${cy}px` }}
    >
      {/* Outer pulse ring */}
      <motion.circle
        cx={cx} cy={cy} r={outer}
        fill="none" stroke={color}
        strokeWidth={isKB ? 1.3 : 1} opacity={isKB ? 0.38 : 0.22}
        animate={{ r: [outer, outer + 7, outer] }}
        transition={{ duration: 2.6 + delay * 0.4, repeat: Infinity, ease: "easeInOut", delay }}
      />
      {/* Crawler: dashed rotating ring */}
      {isHub && (
        <motion.circle
          cx={cx} cy={cy} r={body + 8}
          fill="none" stroke={color} strokeWidth={0.8} strokeOpacity={0.3} strokeDasharray="4 6"
          animate={{ rotate: 360 }}
          transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
          style={{ originX: `${cx}px`, originY: `${cy}px` }}
        />
      )}
      {/* KB: counter-rotating ring */}
      {isKB && (
        <motion.circle
          cx={cx} cy={cy} r={body + 6}
          fill="none" stroke={color} strokeWidth={0.8} strokeOpacity={0.4} strokeDasharray="3 5"
          animate={{ rotate: -360 }}
          transition={{ duration: 6, repeat: Infinity, ease: "linear" }}
          style={{ originX: `${cx}px`, originY: `${cy}px` }}
        />
      )}
      {/* Node body */}
      <circle
        cx={cx} cy={cy} r={body}
        fill={isHub ? "rgba(18,10,55,0.88)" : "rgba(10,15,52,0.85)"}
        stroke={color} strokeWidth={isKB ? 1.8 : 1.3}
        strokeOpacity={isKB ? 1 : isHub ? 0.8 : 0.55}
      />
      {isKB && <circle cx={cx} cy={cy} r={dot + 5} fill={color} opacity={0.18} />}
      {/* Center dot */}
      <circle
        cx={cx} cy={cy} r={dot} fill={color}
        filter={isKB ? "url(#kbGlow)" : isHub ? "url(#hubGlow)" : undefined}
      />
      {/* Label */}
      <text
        x={cx} y={cy + body + 13}
        textAnchor="middle"
        fill="rgba(255,255,255,0.68)"
        style={{ fontSize: "11px", fontFamily: "Inter, sans-serif", fontWeight: 500 }}
      >
        {label}
      </text>
    </motion.g>
  );
}

function DataVisualization() {
  return (
    <svg viewBox="0 0 620 390" className="w-full" style={{ overflow: "visible" }}>
      <defs>
        <filter id="kbGlow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="6" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="hubGlow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="pkgGlow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="softglow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="2.5" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <radialGradient id="hubBg" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="#7c3aed" stopOpacity="0.32" />
          <stop offset="100%" stopColor="#7c3aed" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="kbBg" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="#34d399" stopOpacity="0.18" />
          <stop offset="100%" stopColor="#34d399" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="eg1" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stopColor="#818cf8" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#60a5fa" stopOpacity="0.15" />
        </linearGradient>
        <linearGradient id="eg2" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stopColor="#a78bfa" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#34d399" stopOpacity="0.18" />
        </linearGradient>
      </defs>

      {/* Background glows */}
      <ellipse cx={288} cy={205} rx={155} ry={115} fill="url(#hubBg)" />
      <ellipse cx={520} cy={218} rx={80}  ry={70}  fill="url(#kbBg)" />

      {/* Scatter dots */}
      {SCATTER.map((d, i) => (
        <motion.circle
          key={i} cx={d.cx} cy={d.cy} r={d.r} fill="#c084fc"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0, 0.65, 0.22, 0.65] }}
          transition={{ duration: 3.5 + i * 0.25, delay: d.delay, repeat: Infinity, ease: "easeInOut" }}
        />
      ))}

      {/* Bezier edges */}
      {EDGES.map((e, i) => (
        <motion.path
          key={i} d={e.d} fill="none"
          stroke={i >= 3 ? "url(#eg2)" : "url(#eg1)"}
          strokeWidth="1.3" strokeDasharray="8 15"
          initial={{ opacity: 0, strokeDashoffset: 23 }}
          animate={{ opacity: 0.65, strokeDashoffset: [23, 0] }}
          transition={{
            opacity: { duration: 0.6, delay: e.delay },
            strokeDashoffset: { duration: 2.2, delay: e.delay, repeat: Infinity, ease: "linear" },
          }}
        />
      ))}

      {/* Data packets travelling along edges */}
      {EDGES.map((e, i) => (
        <DataPacket key={i} d={e.d} color={e.packetColor} delay={e.delay + 0.5 + i * 0.28} dur={e.dur} />
      ))}

      {/* Nodes */}
      {NODES.map((n) => (
        <PipelineNode key={n.id} {...n} />
      ))}

      {/* Rising particles */}
      {PARTICLES.map((p, i) => (
        <motion.circle
          key={i} cx={p.cx} r={p.r}
          fill="rgba(167,139,250,0.55)"
          filter="url(#softglow)"
          initial={{ cy: 390, opacity: 0 }}
          animate={{ cy: [390, -10], opacity: [0, 0.65, 0] }}
          transition={{ duration: p.dur, delay: p.delay, repeat: Infinity, ease: "easeIn" }}
        />
      ))}
    </svg>
  );
}

// ── Stat cards ────────────────────────────────────────────────────────────────
const statCards = [
  { icon: TrendingUp, label: "Nguồn tin",       value: "60+",             color: "#818cf8", delay: 0   },
  { icon: Database,   label: "Cơ sở tri thức",  value: "Cập nhật liên tục", color: "#a78bfa", delay: 0.1 },
  { icon: Activity,   label: "Độ chính xác",    value: "Cao",             color: "#34d399", delay: 0.2 },
  { icon: Cpu,        label: "Mô hình AI",      value: "Đa yếu tố",      color: "#f472b6", delay: 0.3 },
];

// ── Input Field component ────────────────────────────────────────────────────
interface InputFieldProps {
  id: string;
  label: string;
  type: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  icon: React.ReactNode;
  rightIcon?: React.ReactNode;
  hasError: boolean;
  autoComplete?: string;
}

function InputField({ id, label, type, value, onChange, placeholder, icon, rightIcon, hasError, autoComplete }: InputFieldProps) {
  const [focused, setFocused] = useState(false);
  return (
    <div>
      <label htmlFor={id} style={{ color: "#64748b", fontSize: "13px", fontWeight: 500, display: "block", marginBottom: "6px" }}>
        {label}
      </label>
      <div
        className="flex items-center gap-3 rounded-xl px-4 transition-all"
        style={{
          background: "#f8fafc",
          border: `1.5px solid ${hasError ? "#ef4444" : focused ? "#6366f1" : "rgba(0,0,0,0.1)"}`,
          boxShadow: focused ? "0 0 0 3px rgba(99,102,241,0.12)" : "none",
          height: "48px",
        }}
      >
        <span style={{ color: focused ? "#6366f1" : "#94a3b8", transition: "color 0.2s", flexShrink: 0 }}>
          {icon}
        </span>
        <input
          id={id}
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={placeholder}
          required
          autoComplete={autoComplete}
          className="flex-1 outline-none bg-transparent"
          style={{ color: "#0f172a", fontSize: "14px" }}
        />
        {rightIcon && <span style={{ flexShrink: 0 }}>{rightIcon}</span>}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function AdminLoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [authState, setAuthState] = useState<AuthState>("idle");
  const [errorMsg, setErrorMsg] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthState("loading");
    setErrorMsg("");

    try {
      const res = await fetch(`${API_URL}/admin/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();

      if (data.success) {
        setAuthState("success");
        setTimeout(() => {
          window.location.href = "http://localhost:3000/dashboard";
        }, 2000);
      } else {
        setAuthState("error");
        setErrorMsg(data.error || "Tên đăng nhập hoặc mật khẩu không đúng.");
      }
    } catch {
      setAuthState("error");
      setErrorMsg("Không thể kết nối đến máy chủ.");
    }
  };

  const gradientBtn = "linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%)";

  return (
    <div className="size-full min-h-screen flex relative overflow-hidden" style={{ fontFamily: "'Inter Tight', 'Inter', sans-serif" }}>

      {/* ── Left brand panel ─────────────────────────────────────────── */}
      <div
        className="relative hidden lg:flex flex-col overflow-hidden"
        style={{
          background: "linear-gradient(135deg, #1e3a8a 0%, #3730a3 30%, #4f46e5 60%, #0e7490 100%)",
          minHeight: "100vh",
          flex: "0 0 48%",
        }}
      >
        {/* Static radial base */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.25) 0%, transparent 55%), radial-gradient(ellipse at 80% 80%, rgba(6,182,212,0.2) 0%, transparent 55%)",
          }}
        />
        {/* Animated grid lines */}
        <motion.div
          className="absolute inset-0 opacity-20"
          animate={{ backgroundPosition: ["0% 0%", "100% 100%", "0% 0%"] }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          style={{
            backgroundImage:
              "repeating-linear-gradient(45deg, transparent, transparent 40px, rgba(255,255,255,0.03) 40px, rgba(255,255,255,0.03) 41px)",
          }}
        />
        {/* Drifting light blobs */}
        <motion.div
          className="absolute pointer-events-none"
          animate={{ x: [0, 50, -30, 0], y: [0, -60, 40, 0] }}
          transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
          style={{ top: "10%", left: "5%", width: 300, height: 300, borderRadius: "50%",
            background: "radial-gradient(circle, rgba(99,102,241,0.22) 0%, transparent 70%)",
            filter: "blur(50px)" }}
        />
        <motion.div
          className="absolute pointer-events-none"
          animate={{ x: [0, -40, 60, 0], y: [0, 50, -30, 0] }}
          transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
          style={{ bottom: "15%", right: "5%", width: 260, height: 260, borderRadius: "50%",
            background: "radial-gradient(circle, rgba(6,182,212,0.2) 0%, transparent 70%)",
            filter: "blur(50px)" }}
        />
        <motion.div
          className="absolute pointer-events-none"
          animate={{ x: [0, 30, -50, 0], y: [0, -30, -60, 0] }}
          transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
          style={{ top: "45%", left: "30%", width: 200, height: 200, borderRadius: "50%",
            background: "radial-gradient(circle, rgba(167,139,250,0.18) 0%, transparent 70%)",
            filter: "blur(40px)" }}
        />

        <div className="relative z-10 flex flex-col h-full px-8 pt-8 pb-6">
          {/* ── Top: logo + brand ── */}
          <motion.div
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="flex items-center gap-3 flex-shrink-0 mb-2"
          >
            <div className="w-9 h-9 rounded-xl flex items-center justify-center overflow-hidden flex-shrink-0"
              style={{ background: "rgba(255,255,255,0.12)", backdropFilter: "blur(8px)" }}>
              <Image src="/LOGO.png" alt="Logo" width={28} height={28} style={{ objectFit: "contain" }} />
            </div>
            <div>
              <div className="text-white font-bold" style={{ fontSize: "15px", letterSpacing: "0.3px" }}>
                Fake Finance<span style={{ color: "#67e8f9" }}>Detector</span>
              </div>
              <div style={{ color: "rgba(255,255,255,0.4)", fontSize: "10px" }}>Enterprise · v1.0</div>
            </div>
          </motion.div>

          {/* ── Middle: graph + title + stats grouped ── */}
          <div className="flex-1 min-h-0 flex flex-col items-center justify-center gap-3">
            <motion.div
              className="w-full"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <DataVisualization />
            </motion.div>

            <motion.div
              className="text-center"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.7 }}
            >
              <h2 className="text-white font-bold" style={{ fontSize: "22px", lineHeight: 1.3, textShadow: "0 2px 20px rgba(0,0,0,0.4)" }}>
                Phát hiện tuyên bố giả
              </h2>
              <h2 style={{ fontSize: "22px", fontWeight: 700, lineHeight: 1.3,
                background: "linear-gradient(90deg, #67e8f9, #818cf8, #60a5fa)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                Tài chính &amp; Tiền mã hoá
              </h2>
              <p style={{ color: "rgba(255,255,255,0.5)", fontSize: "12px", marginTop: "4px" }}>
                Hệ thống AI thời gian thực — phân tích đa nguồn
              </p>
            </motion.div>

            <div className="grid grid-cols-4 gap-2 w-full mb-2">
            {statCards.map((card) => {
              const Icon = card.icon;
              return (
                <motion.div
                  key={card.label}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 1.0 + card.delay }}
                  className="flex flex-col items-center justify-center rounded-xl py-3 px-2 gap-1"
                  style={{
                    background: "rgba(255,255,255,0.07)",
                    backdropFilter: "blur(10px)",
                    border: "1px solid rgba(255,255,255,0.1)",
                  }}
                >
                  <Icon size={16} style={{ color: card.color }} />
                  <div className="text-white font-bold" style={{ fontSize: "13px" }}>{card.value}</div>
                  <div style={{ color: "rgba(255,255,255,0.45)", fontSize: "9px", textAlign: "center" }}>{card.label}</div>
                </motion.div>
              );
            })}
          </div>
          </div>{/* end group */}

          {/* ── Footer ── */}
          <motion.div
            className="flex-shrink-0 flex items-center justify-between"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
          >
            <span style={{ color: "rgba(255,255,255,0.3)", fontSize: "11px" }}>v1.0 · Admin</span>
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span style={{ color: "rgba(255,255,255,0.4)", fontSize: "11px" }}>Hệ thống hoạt động bình thường</span>
            </div>
          </motion.div>
        </div>
      </div>

      {/* ── Right form panel ─────────────────────────────────────────── */}
      <div
        className="flex-1 flex flex-col min-h-screen overflow-y-auto"
        style={{ background: "linear-gradient(135deg, #f0f4ff 0%, #f8faff 100%)" }}
      >
        <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="w-full max-w-[420px]"
          >
            {/* Mobile logo */}
            <div className="lg:hidden flex items-center gap-2 mb-8">
              <Image src="/LOGO.png" alt="Logo" width={32} height={32} style={{ objectFit: "contain" }} />
              <span style={{ color: "#0f172a", fontWeight: 700, fontSize: "16px" }}>
                Fake Finance<span style={{ color: "#6366f1" }}>Detector</span>
              </span>
            </div>

            {/* Card */}
            <div
              className="rounded-2xl p-8"
              style={{
                background: "#ffffff",
                boxShadow: "0 25px 60px rgba(99,102,241,0.12), 0 0 0 1px rgba(99,102,241,0.08)",
              }}
            >
              <AnimatePresence mode="wait">
                {/* ── Success state ── */}
                {authState === "success" ? (
                  <motion.div
                    key="success"
                    initial={{ opacity: 0, scale: 0.85 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex flex-col items-center py-8 text-center"
                  >
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 200, delay: 0.1 }}
                      className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
                      style={{ background: "linear-gradient(135deg, #22c55e, #10b981)" }}
                    >
                      <CheckCircle2 size={32} color="#fff" />
                    </motion.div>
                    <h3 style={{ color: "#0f172a", fontSize: "20px", fontWeight: 700 }}>Đăng nhập thành công!</h3>
                    <p style={{ color: "#64748b", fontSize: "14px", marginTop: "8px" }}>Đang chuyển đến bảng điều khiển…</p>
                    <motion.div className="mt-6 h-1 rounded-full w-full" style={{ background: "#f1f5f9" }}>
                      <motion.div
                        className="h-full rounded-full"
                        style={{ background: gradientBtn }}
                        initial={{ width: "0%" }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 2, ease: "easeInOut" }}
                      />
                    </motion.div>
                  </motion.div>

                ) : (
                  /* ── Login form ── */
                  <motion.div key="login" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div className="mb-6">
                      <h1 style={{ color: "#0f172a", fontSize: "24px", fontWeight: 800 }}>Đăng nhập Admin</h1>
                      <p style={{ color: "#64748b", fontSize: "14px", marginTop: "4px" }}>
                        Truy cập bảng điều khiển quản trị hệ thống
                      </p>
                    </div>

                    {/* Error banner */}
                    <AnimatePresence>
                      {authState === "error" && (
                        <motion.div
                          initial={{ opacity: 0, y: -8, height: 0 }}
                          animate={{ opacity: 1, y: 0, height: "auto" }}
                          exit={{ opacity: 0, y: -8, height: 0 }}
                          className="flex items-start gap-3 rounded-xl px-4 py-3 mb-5"
                          style={{ background: "#fef2f2", border: "1px solid #fecaca" }}
                        >
                          <AlertCircle size={16} style={{ color: "#ef4444", flexShrink: 0, marginTop: "2px" }} />
                          <div>
                            <p style={{ color: "#991b1b", fontSize: "13px", fontWeight: 600 }}>Xác thực thất bại</p>
                            <p style={{ color: "#b91c1c", fontSize: "12px", marginTop: "2px" }}>{errorMsg}</p>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    <form onSubmit={handleSubmit} className="space-y-4">
                      <InputField
                        id="username"
                        label="Tên đăng nhập"
                        type="text"
                        value={username}
                        onChange={setUsername}
                        placeholder="Nhập tên đăng nhập"
                        icon={<User size={16} />}
                        hasError={authState === "error"}
                        autoComplete="username"
                      />
                      <InputField
                        id="password"
                        label="Mật khẩu"
                        type={showPassword ? "text" : "password"}
                        value={password}
                        onChange={setPassword}
                        placeholder="Nhập mật khẩu"
                        icon={<Lock size={16} />}
                        rightIcon={
                          <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            style={{ color: "#94a3b8" }}
                            className="transition-opacity hover:opacity-70"
                          >
                            {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                          </button>
                        }
                        hasError={authState === "error"}
                        autoComplete="current-password"
                      />

                      {/* Submit */}
                      <button
                        type="submit"
                        disabled={authState === "loading"}
                        className="w-full flex items-center justify-center gap-2 rounded-xl py-3.5 transition-all mt-2"
                        style={{
                          background: authState === "loading" ? "#94a3b8" : gradientBtn,
                          color: "#fff",
                          fontWeight: 600,
                          fontSize: "15px",
                          boxShadow: authState === "loading" ? "none" : "0 4px 20px rgba(99,102,241,0.4)",
                          cursor: authState === "loading" ? "not-allowed" : "pointer",
                        }}
                      >
                        {authState === "loading" ? (
                          <>
                            <motion.div
                              className="w-4 h-4 rounded-full border-2 border-white border-t-transparent"
                              animate={{ rotate: 360 }}
                              transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
                            />
                            Đang xác thực…
                          </>
                        ) : (
                          <>
                            Đăng nhập <ArrowRight size={16} />
                          </>
                        )}
                      </button>
                    </form>

                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Status row */}
            <div className="mt-4 flex items-center justify-between px-1">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400" style={{ boxShadow: "0 0 0 3px #dcfce7" }} />
                <span style={{ color: "#22c55e", fontSize: "12px", fontWeight: 500 }}>Hệ thống hoạt động bình thường</span>
              </div>
              <span style={{ color: "#94a3b8", fontSize: "11px" }}>Fake Finance Detector</span>
            </div>
          </motion.div>
        </div>

        <div className="py-4 text-center" style={{ color: "#94a3b8", fontSize: "12px" }}>
          © {new Date().getFullYear()} Fake Finance Claim Detector
        </div>
      </div>
    </div>
  );
}
