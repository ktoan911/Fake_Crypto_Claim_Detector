"use client";

import { useClaimsStats } from "@/src/hooks/useClaimsStats";
import GradientBorderCardV1 from "@/src/components/cards/GradientBorderCardV1";
import { Skeleton } from "@/src/components/ui/skeleton";
import NoData from "@/src/components/status/NoData";
import { useMemo, useState } from "react";
import { ClusterItem } from "@/src/lib/apiClient";

function RateConfirm({ size }: { size: number }) {
  type RateType = "Cao" | "Thấp";
  const rateType: RateType = size > 20 ? "Cao" : "Thấp";
  return (
    <div
      className={`flex items-center justify-center gap-2 border rounded-full px-2 py-1 ${rateType === "Cao" ? "bg-(--rate-high) border-(--text-high)" : "bg-(--rate-low) border-(--text-low)"} min-w-15 max-h-8.5`}
    >
      <p
        className={`font-semibold ${rateType === "Cao" ? "text-(--text-high)" : "text-(--text-low)"}`}
      >
        {rateType}
      </p>
    </div>
  );
}

function CampaignCard({
  item,
}: {
  item: ClusterItem;
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <GradientBorderCardV1 background="var(--card-bg)">
      <div className="relative z-10 py-4 px-5 animate-prediction-rule-card prediction-rule-card-hover h-full flex flex-col">
        <div>
          <div className="flex items-start justify-between mb-2">
            <p className="text-(--text-primary) font-bold text-[1.15rem] leading-snug pe-4">
              {item.topic_label}
            </p>
            <RateConfirm size={item.size} />
          </div>
          <p className="text-(--text-secondary) mb-3 text-[15px]">
            {item.size} tuyên bố liên quan
          </p>
        </div>

        <div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1.5 font-semibold text-[15px] hover:opacity-80 transition-opacity mt-2"
          >
            <span className="bg-clip-text text-transparent bg-(image:--FN-Gradient-2)">
              {isExpanded ? "Ẩn tuyên bố" : "Xem tuyên bố"}
            </span>
            <svg
              className={`w-4 h-4 transition-transform duration-200 text-[#1070e4] dark:text-[#6aabfb] ${isExpanded ? "rotate-180" : ""}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2.5}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>

          <div
            className={`grid transition-all duration-300 ease-in-out ${isExpanded
              ? "grid-rows-[1fr] opacity-100 mt-4"
              : "grid-rows-[0fr] opacity-0 mt-0"
              }`}
          >
            <div className="overflow-hidden">
              <div className="border-t border-border pt-4 flex flex-col gap-3">
                {item.claims.slice(0, 3).map((claim, dIndex) => (
                  <div key={dIndex} className="flex gap-3 items-start">
                    <div>
                      <div className="flex items-start gap-2">
                        <svg
                          className="w-5 h-5 text-(--text-high) shrink-0 mt-[2px]"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                          />
                        </svg>
                        <p className="text-(--text-primary) font-medium text-[14px] leading-snug">
                          {claim}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </GradientBorderCardV1>
  );
}

export default function CardItem() {
  const { clusters, isLoading } = useClaimsStats();
  const topClusters = useMemo(() => {
    if (!Array.isArray(clusters)) return [];
    return [...clusters]
      .sort((a, b) => {
        const sizeA = Number.isFinite(a.size) ? a.size : a.claims.length;
        const sizeB = Number.isFinite(b.size) ? b.size : b.claims.length;
        return sizeB - sizeA;
      })
      .slice(0, 4);
  }, [clusters]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-36 rounded-xl" />
        ))}
      </div>
    );
  }

  if (topClusters.length === 0) {
    return <NoData message="Không có dữ liệu chiến dịch" />;
  }

  return (
    <div className="grid grid-cols-2 gap-4">
      {topClusters.map((item, index) => (
        <CampaignCard key={index} item={item} />
      ))}
    </div>
  );
}
