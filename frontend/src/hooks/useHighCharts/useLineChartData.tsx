import { useMemo, useState } from "react";
import useAreaChartConfig from "./useAreaChartConfig";

import { SeriesOptionsType } from "highcharts";
import { TRange } from "@/src/components/Graph/GraphSection";

// Snapshot type matching the API: { timestamp, fakenew1, fakenew2 }
export type FakeNewSnapshot = {
  timestamp: string;
  fakenew1: number;
  fakenew2: number;
};

export type FakeNewChartInputHookType = {
  rangeConfigs: Array<TRange>;
  snapshots?: FakeNewSnapshot[];
  showMarkers?: boolean;
};

const LINE_COLORS = {
  fakenew1: "#8B5CF6",
  fakenew2: "#10B981",
};

const SERIES_LABELS = {
  fakenew1: "Tổng tuyên bố",
  fakenew2: "Tuyên bố sai",
};

// ── Helpers ────────────────────────────────────────────────────────────────────

function compactNumber(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toFixed(2);
}

export function formatNumber(
  value: number,
  opts: { fractionDigits?: number; prefix?: string } = {},
): string {
  const { fractionDigits = 2, prefix = "" } = opts;
  return `${prefix}${value.toLocaleString("vi-VN", {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  })}`;
}

function formatUTCTimestamp(ts: number, range: TRange): string {
  const d = new Date(ts);
  const days = Number(range.value);
  if (days <= 1) {
    return d.toLocaleTimeString("vi-VN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  }
  if (days <= 7) {
    return d.toLocaleDateString("vi-VN", { month: "short", day: "numeric" });
  }
  return d.toLocaleDateString("vi-VN", { month: "short", day: "numeric" });
}

// ── Hook ───────────────────────────────────────────────────────────────────────

const useLineChartData = ({
  rangeConfigs,
  snapshots,
  showMarkers = false,
}: FakeNewChartInputHookType) => {
  const [range, setRange] = useState(
    rangeConfigs.find((config) => config.value === "7") ||
    rangeConfigs[2] ||
    rangeConfigs[1],
  );

  // Filter snapshots by selected time range
  const filteredSnapshots = useMemo(() => {
    if (!snapshots || snapshots.length === 0) return [];

    const now = new Date();
    const days = Number(range.value);
    const startDate = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);

    return snapshots.filter((snapshot) => {
      const snapshotDate = new Date(snapshot.timestamp);
      return snapshotDate >= startDate;
    });
  }, [snapshots, range]);

  const days = Number(range.value);

  // Build two area series: fakenew1 and fakenew2
  const processedSeries = useMemo(() => {
    return (["fakenew1", "fakenew2"] as const).map((key) => {
      const color = LINE_COLORS[key];
      const data = filteredSnapshots.map(
        (snapshot) =>
          [new Date(snapshot.timestamp).getTime(), snapshot[key]] as [
            number,
            number,
          ],
      );

      return {
        id: key,
        name: SERIES_LABELS[key],
        type: "area" as const,
        data,
        color,
        lineColor: color,
        fillColor: {
          linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
          stops: [
            [0, `${color}33`],
            [1, `${color}00`],
          ] as [number, string][],
        },
        marker: {
          enabled: showMarkers,
          radius: 3,
          fillColor: "#FFFFFF",
          lineWidth: 2,
          lineColor: color,
          symbol: "circle" as const,
          states: {
            hover: {
              enabled: true,
              radius: 5,
              lineWidth: 2,
            },
          },
        },
      };
    });
  }, [filteredSnapshots, showMarkers]);

  const totalDataPoints = useMemo(
    () => processedSeries.reduce((sum, s) => sum + s.data.length, 0),
    [processedSeries],
  );

  const options = useAreaChartConfig(
    {
      chart: {
        height: null,
        reflow: true,
      },
      title: {
        text: "",
      },
      xAxis: {
        type: "datetime",
        crosshair: false,
        labels: {
          formatter: function () {
            return `<p style="color: var(--chart-text1); font-size:12px;">${formatUTCTimestamp(
              this.value as number,
              range,
            )}</p>`;
          },
        },
      },
      yAxis: {
        opposite: false,
        title: {
          text: undefined,
        },
        min: 0,
        crosshair: false,
        gridLineColor: "var(--chart-girdline)",
        gridLineDashStyle: "Dash",
        gridLineWidth: 1,
        labels: {
          formatter() {
            const value = Number(this.value);
            return `<p style="color: var(--chart-text1); font-size:12px;">${compactNumber(value)}</p>`;
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
        itemHoverStyle: {
          color: "var(--primary)",
        },
      },
      tooltip: {
        shared: true,
        enabled: true,
        backgroundColor: "var(--tooltip)",
        borderRadius: 8,
        useHTML: true,
        formatter: function () {
          const dateStr = new Date(this.x as number).toLocaleString("vi-VN", {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          });

          let html = `<div>
            <p style="color: #8D8D8D; font-size: 12px; font-weight: 500; margin-bottom: 8px;">${dateStr}</p>`;

          this.points?.forEach((point) => {
            html += `<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
              <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: ${point.color};"></span>
              <span style="color: #8D8D8D; font-size: 12px;">${point.series.name}:</span>
              <span style="color: #E3E2E7; font-size: 13px; font-weight: 600;">${formatNumber(
              point.y as number,
              { fractionDigits: 2 },
            )}</span>
            </div>`;
          });

          html += "</div>";
          return html;
        },
      },
      plotOptions: {
        area: {
          lineWidth: 2,
          shadow: false,
          states: {
            hover: {
              lineWidth: 2,
            },
          },
        },
      },
      series: processedSeries as unknown as SeriesOptionsType[],
    },
    [processedSeries, range, showMarkers],
  );

  return {
    key: `line-chart-${days}-${totalDataPoints}`,
    options,
    range,
    setRange,
    hasData: filteredSnapshots.length > 0,
    series: processedSeries,
  };
};

export default useLineChartData;
