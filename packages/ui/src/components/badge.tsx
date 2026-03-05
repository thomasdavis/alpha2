import * as React from "react";
import { cn } from "../utils.js";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "outline" | "danger" | "success" | "warning" | "blue";
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  const variants: Record<string, string> = {
    default: "border-transparent bg-accent text-white",
    secondary: "border-transparent bg-surface-2 text-text-primary",
    outline: "text-text-secondary border border-border",
    danger: "border-red/20 bg-red-bg text-red",
    success: "border-green/20 bg-green-bg text-green",
    warning: "border-yellow/20 bg-yellow-bg text-yellow",
    blue: "border-blue/20 bg-blue-bg text-blue",
  };

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-md border px-1.5 py-0.5 text-[0.65rem] font-bold uppercase transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2",
        variants[variant] || variants.default,
        className
      )}
      {...props}
    />
  );
}

export { Badge };
