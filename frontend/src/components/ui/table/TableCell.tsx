/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { cn } from "@/src/lib/utils";
import { TableColumn } from "./types";

interface TableCellProps<T = any> {
  column: TableColumn<T>;
  record: T;
  index: number;
}

const TableCell = <T extends Record<string, any>>({
  column,
  record,
  index,
}: TableCellProps<T>) => {
  // Get the raw value from the record
  const rawValue = column.dataIndex ? record[column.dataIndex] : undefined;

  // Use custom render function if provided, otherwise use the raw value
  const cellContent = column.render
    ? column.render(rawValue, record, index)
    : rawValue;

  const alignClasses = {
    left: "text-left",
    center: "text-center",
    right: "text-right",
  };

  return (
    <td
      className={cn(
        "px-4 py-3",
        alignClasses[column.align || "left"],
        column.cellClassName,
      )}
      style={{
        width: column.width,
        minWidth: column.minWidth,
        maxWidth: column.maxWidth,
      }}
    >
      {cellContent}
    </td>
  );
};

export default TableCell;
