import { ReactNode } from 'react';

// Base column configuration
export interface TableColumn<T = Record<string, unknown>> {
  key: string;
  title: string;
  dataIndex?: keyof T;
  width?: string | number;
  minWidth?: string | number;
  maxWidth?: string | number;
  align?: 'left' | 'center' | 'right';
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: unknown, record: T, index: number) => ReactNode;
  headerRender?: (column: TableColumn<T>) => ReactNode;
  className?: string;
  headerClassName?: string;
  cellClassName?: string;
}

// Table configuration
export interface TableConfig<T = Record<string, unknown>> {
  columns: TableColumn<T>[];
  data: T[];
  loading?: boolean;
  emptyText?: string;
  rowKey?: keyof T | ((record: T) => string | number);
  onRow?: (
    record: T,
    index: number
  ) => {
    onClick?: () => void;
    onDoubleClick?: () => void;
    className?: string;
    style?: React.CSSProperties;
  };
  pagination?: {
    current: number;
    pageSize: number;
    total: number;
    onChange: (page: number, pageSize: number) => void;
    showSizeChanger?: boolean;
    showQuickJumper?: boolean;
  };
  scroll?: {
    x?: string | number;
    y?: string | number;
  };
  size?: 'small' | 'middle' | 'large';
  bordered?: boolean;
  striped?: boolean;
  hoverable?: boolean;
  className?: string;
  headerClassName?: string;
  bodyClassName?: string;
}

// Sorting configuration
export interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}

// Filter configuration
export interface FilterConfig {
  key: string;
  value: unknown;
  operator?:
    | 'eq'
    | 'ne'
    | 'gt'
    | 'lt'
    | 'gte'
    | 'lte'
    | 'contains'
    | 'startsWith'
    | 'endsWith';
}

// Table actions
export interface TableActions<T = Record<string, unknown>> {
  onSort?: (sortConfig: SortConfig) => void;
  onFilter?: (filters: FilterConfig[]) => void;
  onRowSelect?: (selectedRows: T[]) => void;
  onRowExpand?: (record: T, expanded: boolean) => void;
}

// Responsive breakpoints
export type ResponsiveBreakpoint = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';

export interface ResponsiveColumn<T = Record<string, unknown>>
  extends TableColumn<T> {
  responsive?: {
    [K in ResponsiveBreakpoint]?: {
      hide?: boolean;
      width?: string | number;
      align?: 'left' | 'center' | 'right';
    };
  };
}

// Export types for external use
export type { ReactNode };
