"use client";

import Highcharts from "highcharts";
import dynamic from "next/dynamic";
import GeneralCard from "../cards/GeneralCard";
import NoData from "../status/NoData";

export type TRange = {
  id: number;
  title: string;
  value: string;
};

const HighchartsReact = dynamic(() => import("highcharts-react-official"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center py-20">
      <div className="w-10 h-10 border-4 border-[#2A2B2F] border-t-[#a5a5a5] rounded-full animate-spin" />
    </div>
  ),
});

interface ChartCardProps {
  title: string;
  options: Highcharts.Options;
  chartKey: string;
  hasData: boolean;
  isLoading?: boolean;
  rangeConfigs: TRange[];
  range: TRange;
  onRangeChange: (range: TRange) => void;
  noDataMessage?: string;
  className?: string;
}

export default function ChartCard({
  title,
  options,
  chartKey,
  hasData,
  isLoading = false,
  rangeConfigs,
  range,
  onRangeChange,
  noDataMessage = "Không có dữ liệu",
  className,
}: ChartCardProps) {
  return (
    <GeneralCard className={className}>
      <div className="min-w-0 w-full h-full flex flex-col">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-lg font-semibold text-foreground">{title}</h3>
          <div className="flex items-center gap-2">
            {rangeConfigs.map((config) => (
              <button
                key={config.id}
                onClick={() => onRangeChange(config)}
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

        <div className="mt-4 h-[320px] sm:h-[400px] overflow-hidden w-full max-w-full min-w-0 relative">
          {isLoading ? (
            <div className="flex items-center justify-center py-20">
              <div className="w-10 h-10 border-4 border-[#2A2B2F] border-t-[#a5a5a5] rounded-full animate-spin" />
            </div>
          ) : hasData ? (
            <HighchartsReact
              key={chartKey}
              highcharts={Highcharts}
              options={options}
              containerProps={{
                className:
                  "h-full w-full max-w-full [&_.highcharts-container]:!w-full [&_.highcharts-container]:!max-w-full [&_.highcharts-root]:!w-full",
                style: { width: "100%", height: "100%", minWidth: 0 },
              }}
            />
          ) : (
            <NoData message={noDataMessage} />
          )}
        </div>
      </div>
    </GeneralCard>
  );
}
