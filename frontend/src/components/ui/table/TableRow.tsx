"use client";

import React from "react";
import { TableColumn } from "./types";
import TableCell from "./TableCell";
import { cn } from "@/src/lib/utils";

interface TableRowProps<T = unknown> {
  columns: TableColumn<T>[];
  record: T;
  index: number;
  striped?: boolean;
  hoverable?: boolean;
  onClick?: () => void;
  onDoubleClick?: () => void;
  className?: string;
  style?: React.CSSProperties;
}

const TableRow = <T extends Record<string, unknown>>({
  columns,
  record,
  index,
  striped = false,
  hoverable = true,
  onClick,
  onDoubleClick,
  className,
  style,
}: TableRowProps<T>) => {
  const isEven = index % 2 === 0;

  const rowClasses = cn(
    "border-b border-border last:border-b-0",
    striped && isEven && "bg-muted/30",
    hoverable && "hover:bg-muted/50 transition-colors",
    onClick && "cursor-pointer",
    className,
  );

  return (
    <tr
      className={rowClasses}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      style={style}
    >
      {columns.map((column) => (
        <TableCell
          key={column.key}
          column={column}
          record={record}
          index={index}
        />
      ))}
    </tr>
  );
};

export default TableRow;
