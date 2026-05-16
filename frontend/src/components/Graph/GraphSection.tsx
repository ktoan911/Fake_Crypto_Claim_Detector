"use client";

import Highcharts from "highcharts";
import dynamic from "next/dynamic";
import useLineChartData, {
  FakeNewSnapshot,
} from "@/src/hooks/useHighCharts/useLineChartData";
import GeneralCard from "../cards/GeneralCard";
import NoData from "../status/NoData";

export interface TRange {
  id: number;
  title: string;
  value: string;
}

const rangeConfigs: Array<TRange> = [
  {
    id: 0,
    title: "1D",
    value: "1",
  },
  {
    id: 1,
    title: "3D",
    value: "3",
  },
  {
    id: 2,
    title: "7D",
    value: "7",
  },
];

const HighchartsReact = dynamic(() => import("highcharts-react-official"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center py-20">
      <div className="w-10 h-10 border-4 border-[#2A2B2F] border-t-[#a5a5a5] rounded-full animate-spin" />
    </div>
  ),
});
interface GraphSectionProps {
  isLoading?: boolean;
  isSuccess?: boolean;
  snapshots?: FakeNewSnapshot[];
}

export default function GraphSection({
  isLoading,
  isSuccess,
  snapshots,
}: GraphSectionProps) {
  const { key, options, range, setRange, hasData } = useLineChartData({
    rangeConfigs,
    snapshots,
  });

  return (
    <GeneralCard className="mt-4">
      <div>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-lg font-semibold text-foreground">
            Phân Tích Xu Hướng Theo Ngày
          </h3>
          <div className="flex items-center gap-2">
            {rangeConfigs.map((config) => (
              <button
                key={config.id}
                onClick={() => setRange(config)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  range.value === config.value
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {config.title}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-4 h-[400px] sm:h-[400px] overflow-auto w-full relative">
          {isLoading || !isSuccess ? (
            <div className="flex items-center justify-center py-20">
              <div className="w-10 h-10 border-4 border-[#2A2B2F] border-t-[#a5a5a5] rounded-full animate-spin" />
            </div>
          ) : hasData ? (
            <HighchartsReact
              key={key}
              highcharts={Highcharts}
              options={options}
              containerProps={{
                style: { width: "100%", height: "100%" },
              }}
            />
          ) : (
            <NoData message="Không có dữ liệu" />
          )}
        </div>
      </div>
    </GeneralCard>
  );
}
