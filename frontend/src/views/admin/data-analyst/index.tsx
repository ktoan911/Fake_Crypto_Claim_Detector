import IntroAppPage from "@/src/components/IntroAppPage";
import DonutChart from "./DonutChart";
import LineChart from "./LineChart";

export default function DataAnalyst() {
  return (
    <div className="min-w-0">
      <IntroAppPage
        description="Phân tích dữ liệu crawl được hàng ngày"
        title="Phân Tích Dữ Liệu"
      ></IntroAppPage>
      <div className="mt-10 grid min-w-0 grid-cols-1 items-stretch gap-4 md:grid-cols-2">
        <div className="min-w-0 h-full">
          <LineChart />
        </div>
        <div className="min-w-0 h-full">
          <DonutChart />
        </div>
      </div>
    </div>
  );
}
