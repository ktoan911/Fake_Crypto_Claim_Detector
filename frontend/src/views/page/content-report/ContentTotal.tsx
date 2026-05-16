"use client";
import Image from "next/image";

import { useFormatValue } from "@/src/hooks/useFormatValue";
import GradientBorderCardV1 from "@/src/components/cards/GradientBorderCardV1";
import img1 from "@/public/image/img1.png";
import img2 from "@/public/image/img2.png";
import img3 from "@/public/image/img3.png";
import img4 from "@/public/image/img4.png";
import { useTheme } from "next-themes";
import { useClaimsStats } from "@/src/hooks/useClaimsStats";
import { Skeleton } from "@/src/components/ui/skeleton";

const ContentTotal = () => {
  const { resolvedTheme } = useTheme();
  const { isLoading, totalCrawled, totalFake, fakeRate, totalClusters } =
    useClaimsStats();

  const listGeneralInfo = [
    {
      imageSrc: img1.src,
      title: "Tổng tuyên bố hôm nay",
      value: useFormatValue(totalCrawled, 0),
    },
    {
      imageSrc: img2.src,
      title: "Tổng tuyên bố sai đã phát hiện",
      value: useFormatValue(totalFake, 0),
    },
    {
      imageSrc: img3.src,
      title: "Tỷ lệ tuyên bố sai",
      value: useFormatValue(fakeRate, 0) + " %",
    },
    {
      imageSrc: img4.src,
      title: "Số chiến dịch lừa đảo",
      value: useFormatValue(totalClusters, 0),
    },
  ];

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-24 rounded-xl" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
      {listGeneralInfo.map((info, index) => (
        <div key={index}>
          <GradientBorderCardV1 background="var(--card-bg)">
            <div className="flex justify-around items-center animate-leaderboard-card">
              <div className="flex-1/3">
                <Image
                  src={info.imageSrc}
                  alt={info.title}
                  width={100}
                  height={100}
                />
              </div>
              <div className="flex-3/4">
                <h5 className="text-[23px] font-bold bg-clip-text text-transparent bg-(image:--FN-Gradient-1)">
                  {info.value}
                </h5>
                <p
                  className={`${resolvedTheme === "dark" ? "muted" : "text-chart-text1"} text-[15px] font-medium`}
                >
                  {info.title}
                </p>
              </div>
            </div>
          </GradientBorderCardV1>
        </div>
      ))}
    </div>
  );
};

export default ContentTotal;
