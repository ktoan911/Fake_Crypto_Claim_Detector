"use client";
import { useEffect, useState } from "react";
import { Loader2, Search, Database, FileSearch, CheckCircle } from "lucide-react";

interface AnalysisStatusSectionProps {
  isAnalyzing: boolean;
}

const STEPS = [
  { icon: Search, label: "Đang thu thập bằng chứng", duration: 900 },
  { icon: Database, label: "Đang tìm kiếm nguồn tham chiếu", duration: 1400 },
  { icon: FileSearch, label: "Đang đánh giá thông tin", duration: 1800 },
  { icon: CheckCircle, label: "Đang tổng hợp kết luận", duration: 600 },
];

export default function AnalysisStatusSection({
  isAnalyzing,
}: AnalysisStatusSectionProps) {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    if (!isAnalyzing) {
      setCurrentStep(0);
      return;
    }

    let elapsed = 0;
    const timeouts: ReturnType<typeof setTimeout>[] = [];
    STEPS.forEach((step, idx) => {
      elapsed += step.duration;
      timeouts.push(setTimeout(() => setCurrentStep(idx), elapsed));
    });

    return () => timeouts.forEach(clearTimeout);
  }, [isAnalyzing]);

  if (!isAnalyzing) return null;

  return (
    <div
      className="w-full max-w-4xl mx-auto rounded-2xl shadow-lg p-6 md:p-8 border mt-6"
      style={{
        background:
          "linear-gradient(135deg, var(--card-bg) 0%, color-mix(in srgb, var(--card-bg) 80%, #0fb490 20%) 100%)",
        borderColor: "var(--border)",
      }}
    >
      <div>
        <div className="flex items-center gap-3 mb-4">
          <Loader2
            className="w-6 h-6 animate-spin shrink-0"
            style={{ color: "#8b5cf6" }}
          />
          <h3 className="font-semibold">Đang Phân Tích...</h3>
        </div>
        <div className="space-y-3">
            {STEPS.map((step, idx) => {
              const Icon = step.icon;
              const isActive = idx === currentStep;
              const isDone = idx < currentStep;

              return (
                <div key={idx} className="flex items-center gap-3">
                  <div
                    className="p-1.5 rounded-lg transition-all"
                    style={{
                      background: isActive
                        ? "linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)"
                        : isDone
                          ? "linear-gradient(135deg, #06b6d4, #3b82f6)"
                          : "var(--secondary)",
                      transform: isActive ? "scale(1.1)" : "scale(1)",
                    }}
                  >
                    <Icon
                      className="w-4 h-4"
                      style={{
                        color:
                          isActive || isDone
                            ? "white"
                            : "var(--muted-foreground)",
                      }}
                    />
                  </div>
                  <span
                    className="text-sm transition-all"
                    style={{
                      color: isActive
                        ? "#8b5cf6"
                        : isDone
                          ? "#06b6d4"
                          : "var(--muted-foreground)",
                      fontWeight: isActive ? 600 : 400,
                    }}
                  >
                    {step.label}
                  </span>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
}
