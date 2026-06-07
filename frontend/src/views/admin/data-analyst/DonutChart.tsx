"use client";

import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { useTheme } from "next-themes";
import { useMemo, useState } from "react";

import GeneralCard from "@/src/components/cards/GeneralCard";
import NoData from "@/src/components/status/NoData";
import useDonutConfig from "@/src/hooks/useHighCharts/useDonutConfig";

export type DonutChartItem = {
  id?: string;
  name: string;
  value: number;
};

type DonutChartProps = {
  data?: DonutChartItem[];
  isLoading?: boolean;
  isSuccess?: boolean;
};

type ChartPoint = {
  id: string;
  name: string;
  y: number;
  value: number;
  color: Highcharts.ColorType;
};

const fakeData: DonutChartItem[] = [
  { id: "facebook", name: "Facebook", value: 428 },
  { id: "news", name: "Báo điện tử", value: 316 },
  { id: "forum", name: "Diễn đàn", value: 184 },
  { id: "telegram", name: "Telegram", value: 96 },
  { id: "twitter", name: "Twitter", value: 72 },
];

const lightColors = [
  "#3b82f6",
  "#8b5cf6",
  "#06b6d4",
  "#6366f1",
  "#a855f7",
  "#0ea5e9",
  "#7c3aed",
  "#60a5fa",
  "#818cf8",
  "#22d3ee",
];
const darkColors = [
  "#60a5fa",
  "#a78bfa",
  "#22d3ee",
  "#818cf8",
  "#c084fc",
  "#38bdf8",
  "#93c5fd",
  "#7dd3fc",
  "#d8b4fe",
  "#67e8f9",
];

function compactNumber(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toLocaleString("vi-VN");
}

function toPointId(item: DonutChartItem, index: number): string {
  return (
    item.id || item.name.toLowerCase().replace(/\s+/g, "-") || `item-${index}`
  );
}

export default function DonutChart({
  data = fakeData,
  isLoading = false,
  isSuccess = true,
}: DonutChartProps) {
  const { resolvedTheme } = useTheme();
  const [selectedItem, setSelectedItem] = useState<ChartPoint | null>(null);

  const palette = resolvedTheme === "dark" ? darkColors : lightColors;

  const chartData = useMemo<ChartPoint[]>(() => {
    return data
      .map((item, index) => {
        const value = Number.isFinite(item.value) ? item.value : 0;
        const color = palette[index % palette.length];

        return {
          id: toPointId(item, index),
          name: item.name,
          y: value,
          value,
          color,
        };
      })
      .filter((item) => item.value > 0);
  }, [data, palette]);

  const total = useMemo(
    () => chartData.reduce((sum, item) => sum + item.value, 0),
    [chartData],
  );

  const seriesData = useMemo(
    () =>
      chartData.map((item) => ({
        ...item,
        events: {
          mouseOver: () => setSelectedItem(item),
          mouseOut: () => setSelectedItem(null),
        },
      })),
    [chartData],
  );

  const options = useDonutConfig(
    {
      chart: {
        height: 300,
      },
      title: {
        text: "",
      },
      plotOptions: {
        pie: {
          borderRadius: 0,
          states: {
            inactive: {
              enabled: true,
              opacity: 0.2,
            },
          },
        },
      },
      series: [
        {
          name: "Nguồn crawl",
          type: "pie",
          data: seriesData,
        },
      ],
    },
    [seriesData],
  );

  const hasData = total > 0;

  return (
    <GeneralCard className="min-w-0 w-full max-w-full h-full overflow-hidden">
      <div className="min-w-0 w-full h-full flex flex-col">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-lg font-semibold text-foreground">
            Phân Bố Nguồn Crawl
          </h3>
        </div>

        <div className="mt-4 h-[320px] sm:h-[400px] overflow-hidden w-full max-w-full min-w-0 relative">
          {isLoading || !isSuccess ? (
            <div className="flex items-center justify-center py-20">
              <div className="w-10 h-10 border-4 border-[#2A2B2F] border-t-[#a5a5a5] rounded-full animate-spin" />
            </div>
          ) : hasData ? (
            <div className="flex min-w-0 flex-col xl:flex-row items-center justify-center h-full gap-4">
              <div className="relative w-full max-w-full xl:flex-1 h-[210px] sm:h-[280px] flex justify-center items-center min-w-0">
                <HighchartsReact
                  highcharts={Highcharts}
                  options={options}
                  containerProps={{
                    className:
                      "h-full w-full max-w-full [&_.highcharts-container]:!w-full [&_.highcharts-container]:!max-w-full [&_.highcharts-root]:!w-full",
                    style: { width: "100%", height: "100%", minWidth: 0 },
                  }}
                />

                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none text-center">
                  <span className="text-3xl font-bold md:flex text-foreground">
                    {compactNumber(selectedItem ? selectedItem.value : total)}
                  </span>
                  <span className="text-xs text-muted-foreground mt-2 max-w-36">
                    {selectedItem ? selectedItem.name : "Tổng crawl"}
                  </span>
                </div>
              </div>

              <div className="grid min-w-0 grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-3 w-full xl:w-[220px] max-h-[96px] sm:max-h-[120px] xl:max-h-[280px] overflow-y-auto overflow-x-hidden pr-1">
                {chartData.map((item, index) => {
                  const color = palette[index % palette.length];
                  const isSelected = selectedItem?.id === item.id;

                  return (
                    <div
                      key={item.id}
                      className="grid grid-cols-[12px_minmax(0,1fr)_64px] items-start gap-3 rounded-md cursor-pointer transition-all"
                      onMouseEnter={() => setSelectedItem(item)}
                      onMouseLeave={() => setSelectedItem(null)}
                    >
                      <div
                        className="w-3 h-3 rounded-sm shrink-0 mt-1"
                        style={{ background: color }}
                      />
                      <div className="min-w-0">
                        <span
                          className={`text-sm leading-snug font-semibold whitespace-normal break-words transition-colors ${
                            isSelected
                              ? "text-primary"
                              : "text-muted-foreground"
                          }`}
                        >
                          {item.name}
                        </span>
                      </div>
                      <span
                        className={`text-right text-sm font-semibold transition-colors ${
                          isSelected ? "text-primary" : "text-foreground"
                        }`}
                      >
                        {((item.value / total) * 100).toFixed(2)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <NoData message="Không có dữ liệu nguồn crawl" />
          )}
        </div>
      </div>
    </GeneralCard>
  );
}
