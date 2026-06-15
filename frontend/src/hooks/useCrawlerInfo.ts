"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchCrawlerInfo, CrawlerInfo } from "@/src/lib/apiClient";

export function useCrawlerInfo(days: number = 7) {
    return useQuery<CrawlerInfo>({
        queryKey: ["crawler-info", days],
        queryFn: () => fetchCrawlerInfo(days),
        staleTime: 5 * 60 * 1000,
    });
}
