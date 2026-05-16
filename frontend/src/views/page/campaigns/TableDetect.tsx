"use client";
import GeneralCard from "@/src/components/cards/GeneralCard";
import NoData from "@/src/components/status/NoData";
import { Table, TableColumn } from "@/src/components/ui/table";
import { cn } from "@/src/lib/utils";
import { useMemo, useCallback, useRef, useState } from "react";
import { useClaimsStats } from "@/src/hooks/useClaimsStats";
import { Skeleton } from "@/src/components/ui/skeleton";

interface DetectEntry extends Record<string, unknown> {
  id: string;
  claim: string;
  verdict: string;
  checked_at: string;
  confidence: number;
}

const VERDICT_COLOR: Record<string, string> = {
  "Đúng": "bg-(--rate-low) border-(--text-low) text-(--text-low)",
  "Sai": "bg-(--rate-high) border-(--text-high) text-(--text-high)",
  "Chưa chắc chắn": "bg-(--rate-medium) border-(--text-medium) text-(--text-medium)",
};

const TableDetect = () => {
  const tableContainerRef = useRef<HTMLDivElement>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc" | null>(null);
  const { recentClaims, isLoading } = useClaimsStats();

  const data = useMemo<DetectEntry[]>(() => {
    return recentClaims.map((item, index) => ({
      id: index.toString(),
      claim: item.claim,
      verdict: item.verdict,
      checked_at: item.checked_at
        ? new Date(item.checked_at).toLocaleDateString("vi-VN")
        : "—",
      // confidence không có trong recent_claims, dùng 0 fallback
      confidence: 0,
    }));
  }, [recentClaims]);

  const getRowAnimationDelay = useCallback((index: number) => {
    return Math.min(index * 50, 500);
  }, []);

  const displayData = useMemo(() => {
    const result = [...data];
    if (sortDirection === "asc") result.sort((a, b) => a.verdict.localeCompare(b.verdict));
    else if (sortDirection === "desc") result.sort((a, b) => b.verdict.localeCompare(a.verdict));
    return result;
  }, [data, sortDirection]);

  const columns: TableColumn<DetectEntry>[] = [
    {
      key: "claim",
      title: "TUYÊN BỐ",
      dataIndex: "claim",
      width: "55%",
      align: "left",
      render: (value: unknown) => (
        <div className="font-medium text-sm line-clamp-2">{value as string}</div>
      ),
    },
    {
      key: "verdict",
      title: "KẾT LUẬN",
      dataIndex: "verdict",
      width: "20%",
      align: "center",
      headerRender: () => (
        <div
          className="flex items-center justify-center gap-1 cursor-pointer select-none hover:text-(--text-high) transition-colors"
          onClick={() => {
            if (sortDirection === null) setSortDirection("desc");
            else if (sortDirection === "desc") setSortDirection("asc");
            else setSortDirection(null);
          }}
        >
          <span>KẾT LUẬN</span>
          <span className="text-sm font-normal opacity-70">
            {sortDirection === "desc" ? "↓" : sortDirection === "asc" ? "↑" : "↑↓"}
          </span>
        </div>
      ),
      render: (value: unknown) => {
        const v = value as string;
        const cls = VERDICT_COLOR[v] ?? "bg-gray-200 border-gray-400 text-gray-600";
        return (
          <div className={`w-36 mx-auto rounded-full px-2 py-1 border flex items-center justify-center ${cls}`}>
            <div className="font-medium text-sm text-center">{v}</div>
          </div>
        );
      },
    },
    {
      key: "checked_at",
      title: "NGÀY",
      dataIndex: "checked_at",
      width: "15%",
      align: "right",
      render: (value: unknown) => (
        <div className="font-medium text-sm text-gray-400">{value as string}</div>
      ),
    },
  ];

  return (
    <div className="animate-leaderboard-card">
      <GeneralCard>
        <div className="flex flex-col md:flex-row md:items-center justify-between">
          <h5 className="font-bold">Các Tuyên Bố Mới Được Phát Hiện</h5>
        </div>
        {isLoading ? (
          <div className="mt-4 flex flex-col gap-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full rounded" />
            ))}
          </div>
        ) : displayData.length === 0 ? (
          <NoData message="Không có dữ liệu phát hiện" />
        ) : (
          <div ref={tableContainerRef} className="mt-4">
            <Table<DetectEntry>
              columns={columns}
              data={displayData}
              rowKey="id"
              hoverable
              size="small"
              className="bg-transparent"
              onRow={(record, index) => ({
                className: cn(
                  "animate-leaderboard-row leaderboard-row-hover relative",
                ),
                style: {
                  ["--row-delay" as string]: `${getRowAnimationDelay(index)}ms`,
                },
              })}
            />
          </div>
        )}
      </GeneralCard>
    </div>
  );
};

export default TableDetect;
