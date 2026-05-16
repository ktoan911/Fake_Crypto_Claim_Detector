"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/src/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-md font-semibold transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive active:scale-95",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow-xs hover:bg-primary/90",
        destructive:
          "bg-destructive text-white shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60",
        secondary:
          "bg-secondary text-secondary-foreground shadow-xs hover:bg-secondary/80",
        ghost:
          "hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50",
        link: "text-primary underline-offset-4 hover:underline",
        text: "text-foreground bg-secondary hover:bg-card",
        outline1:
          "border bg-background shadow-xs hover:bg-accent hover:text-accent-foreground dark:bg-input/30 dark:border-input dark:hover:bg-input/50 active:scale-95",
        outline2:
          "border border-[#8A3FFC] bg-transparent shadow-xs hover:bg-[#8A3FFC]/10 active:scale-95",
        contained1:
          "bg-[#D5E0FF] text-[#112F61] shadow-xs hover:bg-[#AEC6FF]/90 active:scale-95",
        contained2:
          "bg-[#363B48] text-[#FFFFFF] shadow-xs hover:bg-[#424754] active:scale-95",
        gradient1:
          "bg-background text-foreground shadow-xs hover:opacity-90 hover:bg-[rgba(240, 247, 255, 0.4)] active:scale-95",
        gradient2:
          "bg-gradient-to-r from-green-500 to-blue-500 text-[#D5E0FF] shadow-xs hover:opacity-90 active:scale-95",
        gradient3:
          "bg-background/80 text-foreground shadow-xs disabled:opacity-50 cursor-default",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        xs: "h-5 rounded-sm p-1 text-xs",
        sm: "h-8 rounded-md gap-1.5 px-3 has-[>svg]:px-2.5",
        lg: "h-10 rounded-md px-6 has-[>svg]:px-4",
        icon: "size-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

function Button({
  className,
  variant,
  size,
  asChild = false,
  icon,
  iconPosition = "left",
  iconHidden = false,
  children,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
    icon?: React.ReactNode;
    iconPosition?: "left" | "right";
    iconHidden?: boolean;
  }) {
  const Comp = asChild ? Slot : "button";
  const buttonClassName = cn(
    buttonVariants({ variant, size }),
    iconHidden && "group",
    className,
  );

  const iconContent = icon ? (
    <span
      className={cn(
        "inline-flex h-2 w-2 items-center justify-center shrink-0 transition-all duration-200 ease-out",
        iconHidden
          ? cn(
              "opacity-0 -translate-x-1 scale-75 group-hover:opacity-100 group-hover:translate-x-0 group-hover:scale-100 mr-0",
              iconPosition === "right" && icon
                ? "mr-2.5 ml-[-8]"
                : icon
                  ? "mr-1"
                  : "",
            )
          : iconPosition === "right" && icon
            ? "mr-2 ml-0"
            : icon
              ? "mr-1"
              : "",
      )}
    >
      {icon}
    </span>
  ) : null;

  const iconPlaceholder = icon ? (
    <span
      aria-hidden="true"
      className="inline-flex h-2 w-2 items-center justify-center shrink-0 opacity-0"
    />
  ) : null;

  const content = (
    <div className="flex items-center justify-center gap-3">
      {iconPosition === "left" ? iconContent : iconPlaceholder}
      {children != null ? (
        <span
          className={cn(
            "flex-1 inline-flex items-center justify-center text-center transition-transform duration-200 ease-out",
            iconHidden
              ? iconPosition === "left"
                ? "group-hover:translate-x-2"
                : "group-hover:-translate-x-2"
              : icon && "ml-[-10]",
          )}
        >
          {children}
        </span>
      ) : null}
      {iconPosition === "right" ? iconContent : iconPlaceholder}
    </div>
  );

  // For gradient1 variant, wrap button in a gradient border container
  if (variant === "gradient1") {
    const borderRadiusClass = size === "xs" ? "rounded-sm" : "rounded-md";
    return (
      <div
        className={cn(
          "inline-flex p-px overflow-hidden active:scale-95",
          borderRadiusClass,
        )}
        style={{
          background:
            "linear-gradient(180deg, #FFF 0%, #4DE5C5 30%, #6AABFB 70%, #9133FF 100%)",
        }}
      >
        <Comp
          data-slot="button"
          className={cn(buttonClassName, "w-full")}
          {...props}
        >
          {content}
        </Comp>
      </div>
    );
  }
  if (variant === "gradient3") {
    return (
      <div
        className={cn("inline-flex p-px overflow-hidden rounded-[32px]")}
        style={{
          background:
            "linear-gradient(110deg, #FFF 0%, #9133FF 85%, #6AABFB 40%, #FFF 100%)",
          gap: "10px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Comp
          data-slot="button"
          className={cn(
            buttonVariants({ variant, size, className }),
            "rounded-[32px] py-3 px-6",
          )}
          {...props}
        >
          {content}
        </Comp>
      </div>
    );
  }

  return (
    <Comp data-slot="button" className={buttonClassName} {...props}>
      {content}
    </Comp>
  );
}

export { Button, buttonVariants };
