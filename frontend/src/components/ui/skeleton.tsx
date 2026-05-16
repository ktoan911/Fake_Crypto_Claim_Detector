import { cn } from "@/src/lib/utils";

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md", className)}
      style={{
        background:
          "linear-gradient(90deg, rgba(153, 180, 255, 0.1) 0%, rgba(153, 180, 255, 0.3) 50%, rgba(153, 180, 255, 0.1) 100%)",
      }}
      {...props}
    />
  );
}

export { Skeleton };
