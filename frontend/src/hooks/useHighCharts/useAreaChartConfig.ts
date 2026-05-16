"use client";
import Highcharts, { isArray } from "highcharts";
import { useMemo } from "react";
import useHighchartsDefaultConfig from "./useHighChartsDefaultConfig";

export default function useAreaChartConfig(
  extraOptions: Highcharts.Options,
  deps: unknown[] = [],
) {
  const defaultConfig = useHighchartsDefaultConfig();

  return useMemo(() => {
    return Highcharts.merge(
      defaultConfig,
      {
        chart: {
          type: "area",
        },
        legend: {
          enabled: true,
          layout: "horizontal",
          align: "right",
          verticalAlign: "top",
          itemMarginTop: 0,
          itemMarginBottom: 10,
          itemStyle: {
            color: "#5185AA",
            fontWeight: "500",
          },
        },
        plotOptions: {
          area: {
            marker: {
              enabled: false,
            },
          },
        },
      } as Highcharts.Options,
      {
        ...extraOptions,
        yAxis: isArray(extraOptions.yAxis)
          ? (extraOptions.yAxis as Highcharts.YAxisOptions[]).map((item) => ({
              ...defaultConfig.yAxis,
              ...item,
            }))
          : extraOptions.yAxis,
      },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultConfig, ...deps]);
}
