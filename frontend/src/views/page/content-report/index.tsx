"use client";
import IntroAppPage from "@/src/components/IntroAppPage";
import ContentTotal from "./ContentTotal";
import GraphSection from "@/src/components/Graph/GraphSection";
import DonutChart from "@/src/components/Graph/DonutChart";
import { useClaimsStats } from "@/src/hooks/useClaimsStats";

export default function ContentReport() {
  const { snapshots, isLoading, isSuccess } = useClaimsStats();

  return (
    <>
      <div>
        <IntroAppPage
          title="Báo Cáo Nội Dung"
          description="Thống kê tin tức tài chính / crypto được phát hiện theo ngày và theo nhóm."
        />
        <div className="mt-10">
          <ContentTotal />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-10 gap-4">
          <div className="md:col-span-6">
            <GraphSection
              isLoading={isLoading}
              isSuccess={isSuccess}
              snapshots={snapshots}
            />
          </div>
          <div className="md:col-span-4">
            <DonutChart isLoading={isLoading} isSuccess={isSuccess} />
          </div>
        </div>
      </div>
    </>
  );
}
