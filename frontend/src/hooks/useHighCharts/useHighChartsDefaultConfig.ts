"use client";
import Highcharts from "highcharts";
import { useTheme } from "next-themes";
import { useMemo } from "react";

export const chartColors = ["#8A3FFC", "#BE95FF", "#E8DAFF"];

export default function useHighchartsDefaultConfig(): Highcharts.Options {
  const theme = useTheme();

  return useMemo<Highcharts.Options>(
    () => ({
      chart: {
        backgroundColor: "transparent",
      },
      colors: chartColors,
      title: {
        style: {
          color: theme.resolvedTheme === "dark" ? "#F4F4F4" : "#0A0A0A",
          fontSize: "16px",
        },
      },
      yAxis: {
        opposite: true,
        title: {
          text: "",
        },
        labels: {
          style: {
            color: theme.resolvedTheme === "dark" ? "#F4F4F4" : "#0A0A0A",
            fontFamily: "Inter Tight, Inter Tight Fallback",
          },
        },
        gridLineColor: "rgba(255, 255, 255, 0)",
        lineColor: "transparent",
      },
      xAxis: {
        title: {
          style: {
            color: theme.resolvedTheme === "dark" ? "#F4F4F4" : "#0A0A0A",
            fontFamily: "Inter Tight, Inter Tight Fallback",
          },
        },
        labels: {
          style: {
            color: theme.resolvedTheme === "dark" ? "#F4F4F4" : "#0A0A0A",
            fontFamily: "Inter Tight, Inter Tight Fallback",
          },
        },
        gridLineColor: "rgba(255, 255, 255, 0)",
        lineColor: "transparent",
      },
      tooltip: {
        style: {
          fontFamily: "Inter Tight, Inter Tight Fallback",
        },
      },
      legend: {
        enabled: false,
        itemStyle: {
          color: theme.resolvedTheme === "dark" ? "#F4F4F4" : "#0A0A0A",
          fontWeight: "normal",
          fontSize: "12px",
        },
        itemHoverStyle: {
          color: theme.resolvedTheme === "dark" ? "#F4F4F4" : "#0A0A0A",
        },
      },
      plotOptions: {
        series: {
          marker: {
            enabled: false,
          },
        },
        line: {
          marker: {
            enabled: false,
          },
        },
        area: {
          marker: {
            enabled: false,
          },
        },
      },
      credits: {
        enabled: false,
      },
      loading: {
        style: {
          backgroundColor: theme.resolvedTheme === "dark" ? "#000" : "#fff",
        },
      },
    }),
    [theme],
  );
}
