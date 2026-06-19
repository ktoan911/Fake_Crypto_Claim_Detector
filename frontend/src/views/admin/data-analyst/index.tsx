"use client";

import IntroAppPage from "@/src/components/IntroAppPage";
import { useCrawlerInfo } from "@/src/hooks/useCrawlerInfo";
import DonutChart from "./DonutChart";
import LineChart from "./LineChart";

export default function DataAnalyst() {
  const { data, isLoading } = useCrawlerInfo(7);
  const { data: todayData, isLoading: isTodayLoading, isSuccess: isTodaySuccess } = useCrawlerInfo(1);

  return (
    <div className="min-w-0">
      <IntroAppPage
        description="Phân tích dữ liệu crawl được hàng ngày"
        title="Phân Tích Dữ Liệu"
      ></IntroAppPage>
      <div className="mt-10 grid min-w-0 grid-cols-1 items-stretch gap-4 md:grid-cols-2">
        <div className="min-w-0 h-full">
          <LineChart data={data?.crawl_by_day} isLoading={isLoading} />
        </div>
        <div className="min-w-0 h-full">
          <DonutChart
            data={todayData?.per_source}
            isLoading={isTodayLoading}
            isSuccess={isTodaySuccess}
          />
        </div>
      </div>
    </div>
  );
}
