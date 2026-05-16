/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React from "react";
import { TableColumn } from "./types";
import TableRow from "./TableRow";
import { cn } from "@/src/lib/utils";

interface TableBodyProps<T = any> {
  columns: TableColumn<T>[];
  data: T[];
  getRowKey: (record: T, index: number) => string | number;
  onRow?: (
    record: T,
    index: number,
  ) => {
    onClick?: () => void;
    onDoubleClick?: () => void;
    className?: string;
    style?: React.CSSProperties;
  };
  striped?: boolean;
  hoverable?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

const TableBody = <T extends Record<string, any>>({
  columns,
  data,
  getRowKey,
  onRow,
  striped = false,
  hoverable = true,
  className,
  style,
}: TableBodyProps<T>) => {
  return (
    <tbody className={cn(className)} style={style}>
      {data.map((record, index) => {
        const rowKey = getRowKey(record, index);
        const rowProps = onRow?.(record, index) || {};

        return (
          <TableRow
            key={rowKey}
            columns={columns}
            record={record}
            index={index}
            striped={striped}
            hoverable={hoverable}
            {...rowProps}
          />
        );
      })}
    </tbody>
  );
};

export default TableBody;
