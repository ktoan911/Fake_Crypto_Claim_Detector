import { cn } from "@/src/lib/utils";
import React from "react";

type GeneralCardProps = {
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
};

const GeneralCard = ({ children, className, style }: GeneralCardProps) => {
  return (
    <div
      className={cn(
        "border rounded-lg p-4 flex flex-col gap-3 ",
        "box-custom-blur-border",
        className,
      )}
      style={{ background: "var(--card-bg)", ...style }}
    >
      {children}
    </div>
  );
};

export default GeneralCard;
