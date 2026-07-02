"use client";
import { useState } from "react";
import { Shield } from "lucide-react";
import ClaimInputSection from "./ClaimInputSection";
import AnalysisStatusSection from "./AnalysisStatusSection";
import VerificationResultSection from "./VerificationResultSection";
import { verifyClaim } from "@/src/lib/apiClient";

interface VerifyResult {
  verdict: string;
  status: string;
  evidence?: string[];
  source_links?: string[];
  confidence?: number;
  label_probs?: Record<string, number>;
  error?: string;
}

export default function VerifyView() {
  const [claim, setClaim] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<VerifyResult | null>(null);

  const handleVerify = async () => {
    if (!claim.trim()) return;
    setIsAnalyzing(true);
    setResult(null);

    try {
      const data = await verifyClaim(claim.trim());
      setResult(data);
    } catch (err) {
      setResult({
        verdict: "Lỗi xử lý",
        status: "error",
        error:
          err instanceof Error
            ? err.message
            : "Không thể kết nối đến máy chủ.",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="py-8 px-4">
      {/* ── Centered page header ── */}
      <div className="max-w-4xl mx-auto mb-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div
            className="p-3 rounded-2xl shadow-lg"
            style={{ background: "linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)" }}
          >
            <Shield className="w-8 h-8 text-white" />
          </div>
          <h1
            className="text-4xl font-bold bg-clip-text text-transparent"
            style={{ backgroundImage: "linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)" }}
          >
            Bảng Điều Khiển Xác Minh Sự Thật
          </h1>
        </div>
        <p className="opacity-60 text-sm">
          Xác minh thông tin tài chính / ngân hàng bằng AI với phân tích dựa trên
          bằng chứng thực tế
        </p>
      </div>

      {/* ── Claim input ── */}
      <ClaimInputSection
        claim={claim}
        setClaim={(v) => {
          setClaim(v);
          if (result) setResult(null);
        }}
        onVerify={handleVerify}
        isAnalyzing={isAnalyzing}
      />

      {/* ── Results area ── */}
      <div className="mt-8">
        <AnalysisStatusSection isAnalyzing={isAnalyzing} />

        {result && !isAnalyzing && (
          <VerificationResultSection result={result} />
        )}

        {!result && !isAnalyzing && (
          <div className="max-w-4xl mx-auto flex flex-col items-center justify-center gap-3 py-16 opacity-20">
            <Shield className="w-14 h-14" />
            <p className="text-sm">Nhập tuyên bố để bắt đầu xác minh</p>
          </div>
        )}
      </div>
    </div>
  );
}
