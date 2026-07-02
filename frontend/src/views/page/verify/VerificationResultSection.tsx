"use client";
import { motion } from "motion/react";
import {
  CheckCircle,
  XCircle,
  HelpCircle,
  ExternalLink,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";

interface VerifyApiResponse {
  verdict: string;
  status: string;
  evidence?: string[];
  source_links?: string[];
  confidence?: number;
  label_probs?: Record<string, number>;
  error?: string;
}

interface VerificationResultSectionProps {
  result: VerifyApiResponse;
}

const VERDICT_CONFIG: Record<
  string,
  {
    icon: React.ElementType;
    label: string;
    gradientFrom: string;
    gradientTo: string;
    bg: string;
    border: string;
    text: string;
    description: string;
  }
> = {
  Đúng: {
    icon: CheckCircle,
    label: "Đúng Sự Thật",
    gradientFrom: "#06b6d4",
    gradientTo: "#3b82f6",
    bg: "rgba(6, 182, 212, 0.08)",
    border: "rgba(6, 182, 212, 0.3)",
    text: "#06b6d4",
    description: "Thông tin này được hỗ trợ bởi bằng chứng đáng tin cậy.",
  },
  Sai: {
    icon: XCircle,
    label: "Sai Sự Thật",
    gradientFrom: "#e05252",
    gradientTo: "#c0392b",
    bg: "rgba(224, 82, 82, 0.08)",
    border: "rgba(224, 82, 82, 0.3)",
    text: "#e05252",
    description: "Thông tin này mâu thuẫn với bằng chứng đã xác minh.",
  },
  "Chưa chắc chắn": {
    icon: HelpCircle,
    label: "Chưa Chắc Chắn",
    gradientFrom: "#f39c12",
    gradientTo: "#e67e22",
    bg: "rgba(243, 156, 18, 0.08)",
    border: "rgba(243, 156, 18, 0.3)",
    text: "#f39c12",
    description: "Không đủ bằng chứng để xác minh chắc chắn thông tin này.",
  },
  "Lỗi xử lý": {
    icon: XCircle,
    label: "Lỗi Xử Lý",
    gradientFrom: "#95a5a6",
    gradientTo: "#7f8c8d",
    bg: "rgba(149, 165, 166, 0.08)",
    border: "rgba(149, 165, 166, 0.3)",
    text: "#95a5a6",
    description: "Đã xảy ra lỗi khi xử lý yêu cầu.",
  },
};

function getVerdictConfig(verdict: string) {
  return (
    VERDICT_CONFIG[verdict] ?? {
      icon: HelpCircle,
      label: verdict,
      gradientFrom: "#8b5cf6",
      gradientTo: "#3b82f6",
      bg: "rgba(105, 19, 203, 0.08)",
      border: "rgba(105, 19, 203, 0.3)",
      text: "#8b5cf6",
      description: "Kết quả xác minh từ hệ thống AI.",
    }
  );
}

function ConfidenceBar({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const isHigh = pct >= 75;
  const isMid = pct >= 50;

  const color = isHigh
    ? "linear-gradient(90deg, #06b6d4, #3b82f6)"
    : isMid
      ? "linear-gradient(90deg, #f39c12, #e67e22)"
      : "linear-gradient(90deg, #e05252, #c0392b)";

  const Icon = isHigh ? TrendingUp : isMid ? Minus : TrendingDown;
  const label = isHigh
    ? "Độ Tin Cậy Cao"
    : isMid
      ? "Độ Tin Cậy Trung Bình"
      : "Độ Tin Cậy Thấp";
  const textColor = isHigh ? "#06b6d4" : isMid ? "#f39c12" : "#e05252";

  return (
    <div
      className="rounded-2xl shadow-md p-6 border mt-4"
      style={{ backgroundColor: "var(--card-bg)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div
            className="p-2 rounded-xl"
            style={{ background: color }}
          >
            <Icon className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="font-semibold text-sm">Điểm Tin Cậy</p>
            <p className="text-xs" style={{ color: textColor }}>
              {label}
            </p>
          </div>
        </div>
        <span
          className="text-3xl font-bold bg-clip-text text-transparent"
          style={{ backgroundImage: "linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)" }}
        >
          {pct}%
        </span>
      </div>

      <div
        className="relative h-2.5 rounded-full overflow-hidden"
        style={{ backgroundColor: "var(--secondary)" }}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>

      <p className="text-xs opacity-50 mt-3">
        {isHigh
          ? "Bằng chứng mạnh mẽ hỗ trợ đánh giá này"
          : isMid
            ? "Có bằng chứng vừa phải cho đánh giá này"
            : "Bằng chứng hạn chế — cần thêm thông tin xác minh"}
      </p>
    </div>
  );
}


const LABEL_ORDER = ["Đúng", "Sai", "Chưa chắc chắn"];

function LabelProbsSection({
  labelProbs,
  activeVerdict,
}: {
  labelProbs: Record<string, number>;
  activeVerdict: string;
}) {
  const labels = LABEL_ORDER.filter((label) => label in labelProbs);

  return (
    <div
      className="rounded-2xl shadow-md p-6 border mt-4"
      style={{ backgroundColor: "var(--card-bg)", borderColor: "var(--border)" }}
    >
      <p className="font-semibold text-sm mb-4">Phân Bố Xác Suất Theo Nhãn</p>
      <div className="space-y-4">
        {labels.map((label) => {
          const cfg = getVerdictConfig(label);
          const pct = Math.round(labelProbs[label] * 100);
          const isActive = label === activeVerdict;

          return (
            <div key={label}>
              <div className="flex items-center justify-between mb-1.5">
                <span
                  className="text-sm"
                  style={{
                    color: isActive ? cfg.text : "var(--foreground)",
                    fontWeight: isActive ? 600 : 400,
                    opacity: isActive ? 1 : 0.7,
                  }}
                >
                  {cfg.label}
                </span>
                <span
                  className="text-sm font-semibold"
                  style={{ color: isActive ? cfg.text : "var(--foreground)", opacity: isActive ? 1 : 0.7 }}
                >
                  {pct}%
                </span>
              </div>
              <div
                className="relative h-2 rounded-full overflow-hidden"
                style={{ backgroundColor: "var(--secondary)" }}
              >
                <div
                  className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
                  style={{
                    width: `${pct}%`,
                    background: `linear-gradient(90deg, ${cfg.gradientFrom}, ${cfg.gradientTo})`,
                    opacity: isActive ? 1 : 0.5,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SourceLinksSection({ links }: { links: string[] }) {
  if (!links || links.length === 0) return null;

  return (
    <div
      className="rounded-2xl shadow-md p-6 border mt-4"
      style={{ backgroundColor: "var(--card-bg)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center gap-2 mb-4">
        <ExternalLink className="w-5 h-5" style={{ color: "#06b6d4" }} />
        <h3 className="font-semibold">Nguồn Tham Chiếu</h3>
      </div>

      <div className="space-y-2">
        {links.map((link, idx) => (
          <a
            key={idx}
            href={link}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 p-3 rounded-xl text-sm transition-opacity hover:opacity-80 border"
            style={{
              backgroundColor: "var(--secondary)",
              borderColor: "var(--border)",
              color: "#3b82f6",
            }}
          >
            <ExternalLink className="w-3.5 h-3.5 shrink-0" />
            <span className="truncate">{link}</span>
          </a>
        ))}
      </div>
    </div>
  );
}

export default function VerificationResultSection({
  result,
}: VerificationResultSectionProps) {
  const cfg = getVerdictConfig(result.verdict);
  const Icon = cfg.icon;

  return (
    <motion.div
      className="mt-6 w-full max-w-4xl mx-auto"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Verdict Card */}
      <div
        className="rounded-2xl shadow-xl p-6 md:p-8 border"
        style={{ backgroundColor: cfg.bg, borderColor: cfg.border }}
      >
        <div className="flex items-center gap-5">
          <motion.div
            className="p-4 rounded-2xl shadow-lg shrink-0"
            style={{
              background: `linear-gradient(135deg, ${cfg.gradientFrom}, ${cfg.gradientTo})`,
            }}
            initial={{ scale: 0.7, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.1, duration: 0.35, type: "spring" }}
          >
            <Icon className="w-10 h-10 text-white" />
          </motion.div>
          <div className="flex-1 min-w-0">
            <h2
              className="text-2xl md:text-3xl font-bold mb-1"
              style={{ color: cfg.text }}
            >
              {cfg.label}
            </h2>
            <p className="text-sm opacity-70">{cfg.description}</p>
            {result.error && (
              <p className="text-xs mt-2 opacity-50 truncate">{result.error}</p>
            )}
          </div>
        </div>
      </div>

      {/* Label probability breakdown (all 3 nhãn kèm phần trăm) */}
      {result.status === "success" && result.label_probs ? (
        <LabelProbsSection
          labelProbs={result.label_probs}
          activeVerdict={result.verdict}
        />
      ) : (
        result.confidence !== undefined &&
        result.status === "success" && (
          <ConfidenceBar confidence={result.confidence} />
        )
      )}

      {/* Sources */}
      {result.source_links && result.source_links.length > 0 && (
        <SourceLinksSection links={result.source_links} />
      )}
    </motion.div>
  );
}
