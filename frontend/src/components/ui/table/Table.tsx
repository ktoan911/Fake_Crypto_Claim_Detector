"use client";

import React, { useMemo } from "react";
import { TableConfig } from "./types";
import TableHeader from "./TableHeader";
import TableBody from "./TableBody";
import { Database } from "lucide-react";
import { cn } from "@/src/lib/utils";

interface TableProps<T = unknown> extends TableConfig<T> {
  className?: string;
  style?: React.CSSProperties;
}

const Table = <T extends Record<string, unknown>>({
  columns,
  data,
  loading = false,
  // emptyText = "No data available",
  rowKey = "id",
  onRow,
  // pagination,
  scroll,
  size = "middle",
  bordered = false,
  striped = false,
  hoverable = true,
  className,
  headerClassName,
  bodyClassName,
  ...props
}: TableProps<T>) => {
  // Generate unique row keys
  const getRowKey = (record: T, index: number): string | number => {
    if (typeof rowKey === "function") {
      return rowKey(record);
    }
    return (record[rowKey] as string | number) || index;
  };

  // Size-based styling
  const sizeClasses = {
    small: "text-xs",
    middle: "text-sm",
    large: "text-base",
  };

  // Table container classes
  const tableContainerClasses = cn(
    "relative overflow-hidden",
    scroll?.x && "overflow-x-auto",
    scroll?.y && "overflow-y-auto",
    className,
  );

  // Table classes
  const tableClasses = cn(
    "w-full border-collapse",
    sizeClasses[size],
    bordered && "border border-border",
    "min-w-full",
  );

  // Handle scroll styles
  const scrollStyles = useMemo(() => {
    const styles: React.CSSProperties = {};
    if (scroll?.x) {
      styles.minWidth =
        typeof scroll.x === "number" ? `${scroll.x}px` : scroll.x;
    }
    if (scroll?.y) {
      styles.maxHeight =
        typeof scroll.y === "number" ? `${scroll.y}px` : scroll.y;
    }
    return styles;
  }, [scroll]);

  // Loading state
  if (loading) {
    return (
      <div className={tableContainerClasses}>
        <div className="flex items-center justify-center h-32">
          <div className="text-muted-foreground">Đang tải...</div>
        </div>
      </div>
    );
  }

  // Empty state
  if (!data || data.length === 0) {
    return (
      <div className={tableContainerClasses}>
        <div className="flex items-center justify-center h-32">
          <div className="text-muted-foreground">
            {
              <>
                <Database className="w-10 h-10 text-muted-foreground" />
              </>
            }
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={tableContainerClasses} style={scrollStyles}>
      <table className={tableClasses} {...props}>
        <TableHeader columns={columns} className={headerClassName} />
        <TableBody
          columns={columns}
          data={data}
          getRowKey={getRowKey}
          onRow={onRow}
          striped={striped}
          hoverable={hoverable}
          className={bodyClassName}
        />
      </table>
    </div>
  );
};

export default Table;
