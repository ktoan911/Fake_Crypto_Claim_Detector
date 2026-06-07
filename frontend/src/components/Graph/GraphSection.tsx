"use client";

import useLineChartData, {
  FakeNewSnapshot,
} from "@/src/hooks/useHighCharts/useLineChartData";
import ChartCard, { TRange } from "./ChartCard";

export type { TRange };

const rangeConfigs: Array<TRange> = [
  { id: 0, title: "1D", value: "1" },
  { id: 1, title: "3D", value: "3" },
  { id: 2, title: "7D", value: "7" },
];

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
    <ChartCard
      title="Phân Tích Xu Hướng Theo Ngày"
      options={options}
      chartKey={key}
      hasData={hasData}
      isLoading={isLoading || !isSuccess}
      rangeConfigs={rangeConfigs}
      range={range}
      onRangeChange={setRange}
      className="mt-4"
    />
  );
}
