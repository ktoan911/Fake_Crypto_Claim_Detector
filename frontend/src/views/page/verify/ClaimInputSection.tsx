"use client";
import { Sparkles, Eraser, FileText } from "lucide-react";

interface ClaimInputSectionProps {
  claim: string;
  setClaim: (claim: string) => void;
  onVerify: () => void;
  isAnalyzing: boolean;
}

const EXAMPLE_CLAIMS = [
  "Giá vàng sẽ tăng 20% trong tháng tới do biến động thị trường.",
  "Công ty XYZ đã đạt được doanh thu kỷ lục trong quý vừa qua.",
  "Ngân hàng ABC sắp phá sản vì nợ xấu tăng cao."
];

export default function ClaimInputSection({
  claim,
  setClaim,
  onVerify,
  isAnalyzing,
}: ClaimInputSectionProps) {
  return (
    <div className="w-full max-w-4xl mx-auto">
      <div
        className="rounded-2xl shadow-lg p-8 border"
        style={{ backgroundColor: "var(--card-bg)", borderColor: "var(--border)" }}
      >
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div
            className="p-2 rounded-xl"
            style={{ background: "linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)" }}
          >
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold">Xác Minh Thông Tin</h2>
            <p className="text-sm opacity-60">
              Nhập bất kỳ tuyên bố hoặc thông tin cần kiểm tra
            </p>
          </div>
        </div>

        {/* Textarea */}
        <textarea
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          placeholder="Nhập thông tin, tuyên bố hoặc nội dung tin tức tài chính cần xác minh..."
          className="w-full min-h-[160px] p-4 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all text-sm"
          style={{
            backgroundColor: "var(--input)",
            color: "var(--foreground)",
            border: "1px solid var(--border)",
          }}
          disabled={isAnalyzing}
        />

        {/* Actions */}
        <div className="flex items-center justify-between mt-4">
          <button
            onClick={() => setClaim("")}
            disabled={isAnalyzing || !claim}
            className="px-4 py-2 rounded-xl transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 text-sm"
            style={{
              backgroundColor: "var(--secondary)",
              color: "var(--secondary-foreground)",
            }}
          >
            <Eraser className="w-4 h-4" />
            Xóa
          </button>

          <button
            onClick={onVerify}
            disabled={isAnalyzing || !claim.trim()}
            className="px-6 py-2.5 rounded-xl text-white text-sm font-semibold transition-all hover:opacity-90 hover:shadow-lg disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
            style={{ background: "linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)" }}
          >
            <Sparkles className="w-4 h-4" />
            {isAnalyzing ? "Đang Xác Minh..." : "Xác Minh"}
          </button>
        </div>

        {/* Example claims */}
        {!claim && (
          <div
            className="mt-6 pt-6 border-t"
            style={{ borderColor: "var(--border)" }}
          >
            <p className="text-sm opacity-50 mb-3">Thử các ví dụ sau:</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_CLAIMS.map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => setClaim(example)}
                  className="px-3 py-1.5 text-sm rounded-lg transition-opacity hover:opacity-70 flex items-center gap-1.5"
                  style={{
                    backgroundColor: "var(--secondary)",
                    color: "var(--foreground)",
                  }}
                >
                  <FileText className="w-3.5 h-3.5 shrink-0 opacity-60" />
                  {example.substring(0, 50)}...
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
