"use client";

/**
 * Format a number value with thousands separators and decimal places
 * @param value - The number or string to format
 * @param decimals - Number of decimal places (default: 2)
 * @param fallback - Value to return if input is invalid (default: '-')
 * @returns Formatted number string (e.g., "1,234.56")
 *
 * @example
 * formatValue(1234.567) // "1,234.57"
 * formatValue(1234.567, 0) // "1,235"
 * formatValue(1234.567, 4) // "1,234.5670"
 * formatValue(null) // "-"
 * formatValue(null, 2, 'N/A') // "N/A"
 */
export const formatValue = (
  value: number | string | null | undefined,
  decimals?: number,
  fallback?: string,
): string => {
  if (value === null || value === undefined || value === "") {
    return fallback || "-";
  }

  const numValue = typeof value === "string" ? parseFloat(value) : value;

  if (isNaN(numValue)) {
    return fallback || "-";
  }
  return numValue.toLocaleString("vi-VN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

/**
 * Custom hook to format a number value
 * This is a wrapper around formatValue for use in React components
 *
 * @param value - The number or string to format
 * @param decimals - Number of decimal places (default: 2)
 * @param fallback - Value to return if input is invalid (default: '-')
 * @returns Formatted number string
 *
 * @example
 * const formattedValue = useFormatValue(1234.567);
 * // Returns: "1,234.57"
 *
 * const formattedValue = useFormatValue(data?.amount, 2, 'N/A');
 * // Returns: "1,234.57" or "N/A" if data is null
 */
export const useFormatValue = (
  value: number | string | null | undefined,
  decimals?: number,
  fallback?: string,
): string => {
  return formatValue(value, decimals, fallback);
};
