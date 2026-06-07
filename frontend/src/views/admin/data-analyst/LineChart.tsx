"use client";

import { useMemo, useState } from "react";
import { SeriesOptionsType } from "highcharts";
import ChartCard, { TRange } from "@/src/components/Graph/ChartCard";
import useAreaChartConfig from "@/src/hooks/useHighCharts/useAreaChartConfig";

export type CrawlByDay = {
  total_crawl: number;
  day: string;
};

type LineChartProps = {
  data?: CrawlByDay[];
  isLoading?: boolean;
};

const rangeConfigs: TRange[] = [
  { id: 1, title: "3D", value: "3" },
  { id: 2, title: "7D", value: "7" },
];

const fakeCrawlData: CrawlByDay[] = [
  { day: "2026-05-24", total_crawl: 128 },
  { day: "2026-05-25", total_crawl: 164 },
  { day: "2026-05-26", total_crawl: 142 },
  { day: "2026-05-27", total_crawl: 196 },
  { day: "2026-05-28", total_crawl: 238 },
  { day: "2026-05-29", total_crawl: 214 },
  { day: "2026-05-30", total_crawl: 276 },
  { day: "2026-05-31", total_crawl: 251 },
  { day: "2026-06-01", total_crawl: 304 },
  { day: "2026-06-02", total_crawl: 332 },
  { day: "2026-06-03", total_crawl: 298 },
  { day: "2026-06-04", total_crawl: 356 },
  { day: "2026-06-05", total_crawl: 389 },
  { day: "2026-06-06", total_crawl: 421 },
];

function compactNumber(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toLocaleString("vi-VN");
}

function formatDay(timestamp: number): string {
  return new Date(timestamp).toLocaleDateString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
  });
}

export default function LineChart({
  data = fakeCrawlData,
  isLoading = false,
}: LineChartProps) {
  const [range, setRange] = useState(rangeConfigs[0]);

  const filteredData = useMemo(() => {
    const sortedData = [...data].sort(
      (a, b) => new Date(a.day).getTime() - new Date(b.day).getTime(),
    );
    return sortedData.slice(-Number(range.value));
  }, [data, range]);

  const seriesData = useMemo(
    () =>
      filteredData.map(
        (item) =>
          [new Date(`${item.day}T00:00:00`).getTime(), item.total_crawl] as [
            number,
            number,
          ],
      ),
    [filteredData],
  );

  const options = useAreaChartConfig(
    {
      chart: { height: null, reflow: true },
      title: { text: "" },
      xAxis: {
        type: "datetime",
        crosshair: false,
        labels: {
          formatter() {
            return `<p style="color: var(--chart-text1); font-size:12px;">${formatDay(this.value as number)}</p>`;
          },
        },
      },
      yAxis: {
        opposite: false,
        title: { text: undefined },
        min: 0,
        crosshair: false,
        gridLineColor: "var(--chart-girdline)",
        gridLineDashStyle: "Dash",
        gridLineWidth: 1,
        labels: {
          formatter() {
            return `<p style="color: var(--chart-text1); font-size:12px;">${compactNumber(Number(this.value))}</p>`;
          },
        },
      },
      legend: {
        enabled: true,
        align: "center",
        verticalAlign: "bottom",
        itemStyle: {
          color: "var(--secondary-foreground)",
          fontSize: "14px",
          fontWeight: "500",
        },
        itemHoverStyle: { color: "var(--primary)" },
      },
      tooltip: {
        shared: true,
        enabled: true,
        backgroundColor: "var(--tooltip)",
        borderRadius: 8,
        useHTML: true,
        formatter() {
          const day = new Date(this.x as number).toLocaleDateString("vi-VN", {
            weekday: "short",
            day: "2-digit",
            month: "2-digit",
            year: "numeric",
          });
          return `<div>
            <p style="color:#8D8D8D;font-size:12px;font-weight:500;margin-bottom:8px;">${day}</p>
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#6913cb;"></span>
              <span style="color:#8D8D8D;font-size:12px;">Số bài viết crawl:</span>
              <span style="color:#6913cb;font-size:13px;font-weight:600;">${compactNumber(Number(this.y))}</span>
            </div>
          </div>`;
        },
      },
      plotOptions: {
        area: {
          lineWidth: 2,
          shadow: false,
          states: { hover: { lineWidth: 2 } },
        },
      },
      series: [
        {
          id: "total-crawl",
          name: "Số bài viết crawl",
          type: "area",
          data: seriesData,
          color: "#6913cb",
          lineColor: "#6913cb",
          fillColor: {
            linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
            stops: [
              [0, "#c43bf633"],
              [1, "#3B82F600"],
            ],
          },
          marker: {
            enabled: false,
            radius: 3,
            fillColor: "#FFFFFF",
            lineWidth: 2,
            lineColor: "#6913cb",
            symbol: "circle",
            states: { hover: { enabled: true, radius: 5, lineWidth: 2 } },
          },
        },
      ] as SeriesOptionsType[],
    },
    [seriesData, range],
  );

  return (
    <ChartCard
      title="Số Lượng Bài Viết Crawl"
      options={options}
      chartKey={`crawl-line-chart-${range.value}-${seriesData.length}`}
      hasData={seriesData.length > 0}
      isLoading={isLoading}
      rangeConfigs={rangeConfigs}
      range={range}
      onRangeChange={setRange}
      noDataMessage="Không có dữ liệu crawl"
    />
  );
}
