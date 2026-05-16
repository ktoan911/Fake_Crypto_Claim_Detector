"use client";

import React from "react";
import { TableColumn } from "./types";
import { cn } from "@/src/lib/utils";

interface TableHeaderProps<T = unknown> {
  columns: TableColumn<T>[];
  className?: string;
  style?: React.CSSProperties;
}

const TableHeader = <T extends Record<string, unknown>>({
  columns,
  className,
  style,
}: TableHeaderProps<T>) => {
  return (
    <thead className={cn("bg-muted/50", className)} style={style}>
      <tr>
        {columns.map((column) => {
          const headerContent = column.headerRender
            ? column.headerRender(column)
            : column.title;

          const alignClasses = {
            left: "text-left",
            center: "text-center",
            right: "text-right",
          };

          return (
            <th
              key={column.key}
              className={cn(
                "px-4 py-3 font-semibold text-muted-foreground",
                "border-b border-border",
                alignClasses[column.align || "left"],
                column.headerClassName,
                column.sortable &&
                  "cursor-pointer hover:bg-muted/70 select-none",
                "whitespace-nowrap",
              )}
              style={{
                width: column.width,
                minWidth: column.minWidth,
                maxWidth: column.maxWidth,
              }}
            >
              {headerContent}
            </th>
          );
        })}
      </tr>
    </thead>
  );
};

export default TableHeader;
