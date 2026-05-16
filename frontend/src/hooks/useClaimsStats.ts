"use client";

import { useQuery } from "@tanstack/react-query";
import {
    fetchClaimsStats,
    ClaimsStats,
    ClusterItem,
    RawClusterItem,
    RecentClaim,
} from "@/src/lib/apiClient";
import { FakeNewSnapshot } from "@/src/hooks/useHighCharts/useLineChartData";

/** Convert daily_total + daily_false maps → GraphSection snapshots array */
function buildSnapshots(
    daily_total: Record<string, number>,
    daily_false: Record<string, number>
): FakeNewSnapshot[] {
    const dates = Object.keys(daily_total).sort();
    return dates.map((date) => ({
        timestamp: new Date(date + "T12:00:00Z").toISOString(),
        fakenew1: daily_total[date] ?? 0,  // total claims
        fakenew2: daily_false[date] ?? 0,  // false claims
    }));
}

function getRawClusters(clusterData: ClaimsStats["cluster"] | undefined): RawClusterItem[] {
    if (Array.isArray(clusterData)) return clusterData as RawClusterItem[];

    if (
        clusterData &&
        typeof clusterData === "object" &&
        "clusters" in clusterData &&
        Array.isArray((clusterData as { clusters?: unknown }).clusters)
    ) {
        return (clusterData as { clusters: RawClusterItem[] }).clusters;
    }

    return [];
}

function normalizeClusters(clusterData: ClaimsStats["cluster"] | undefined): ClusterItem[] {
    const rawClusters = getRawClusters(clusterData);

    return rawClusters
        .map((item, index) => {
            const claims = Array.isArray(item.claims)
                ? item.claims.filter((c): c is string => typeof c === "string" && c.trim().length > 0)
                : [];

            const size = typeof item.size === "number" && Number.isFinite(item.size)
                ? item.size
                : claims.length;

            const topicLabelCandidates = [
                item.topic_label,
                item.cluster_content,
                item.representative_claim,
            ]
                .map((v) => (typeof v === "string" ? v.trim() : ""))
                .filter(Boolean);

            return {
                topic_label: topicLabelCandidates[0] ?? `Cluster ${index + 1}`,
                claims,
                size,
                representative_claim: item.representative_claim,
                cluster_content: item.cluster_content,
                cluster_id: item.cluster_id,
            };
        })
        .filter((item) => item.size > 0 || item.claims.length > 0);
}

export interface UseClaimsStatsReturn {
    data: ClaimsStats | undefined;
    isLoading: boolean;
    isError: boolean;
    isSuccess: boolean;
    // Derived
    snapshots: FakeNewSnapshot[];
    totalCrawled: number;
    totalFake: number;
    fakeRate: number;
    totalClusters: number;
    clusters: ClusterItem[];
    recentClaims: RecentClaim[];
    realCount: number;
    uncertainCount: number;
}

export function useClaimsStats(date?: string): UseClaimsStatsReturn {
    const { data, isLoading, isError, isSuccess } = useQuery({
        queryKey: ["claimsStats", date ?? "today"],
        queryFn: () => fetchClaimsStats(date),
        staleTime: 5 * 60 * 1000, // 5 phút
    });

    const snapshots = data
        ? buildSnapshots(data.daily_total ?? {}, data.daily_false ?? {})
        : [];

    const stats = data?.stats_24h;
    const totalCrawled = stats
        ? (stats["đúng"] ?? 0) + (stats["sai"] ?? 0) + (stats["chưa chắc chắn"] ?? 0)
        : 0;
    const totalFake = stats?.["sai"] ?? 0;
    const realCount = stats?.["đúng"] ?? 0;
    const uncertainCount = stats?.["chưa chắc chắn"] ?? 0;
    const fakeRate = totalCrawled > 0 ? Math.round((totalFake / totalCrawled) * 100) : 0;
    const clusters = normalizeClusters(data?.cluster);
    const recentClaims = data?.recent_claims ?? [];

    return {
        data,
        isLoading,
        isError,
        isSuccess,
        snapshots,
        totalCrawled,
        totalFake,
        fakeRate,
        totalClusters: clusters.length,
        clusters,
        recentClaims,
        realCount,
        uncertainCount,
    };
}
