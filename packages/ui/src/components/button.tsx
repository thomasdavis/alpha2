import * as React from "react";
import { cn } from "../utils.js";
import { Spinner } from "./spinner.js";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
  size?: "sm" | "md" | "lg" | "icon";
  loading?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", loading, children, ...props }, ref) => {
    const variants: Record<string, string> = {
      primary: "bg-accent text-white hover:bg-accent/90 shadow-sm",
      secondary: "bg-surface-2 text-text-primary hover:bg-surface-2/80 border border-border shadow-sm",
      outline: "border border-border bg-transparent hover:bg-surface-2 text-text-secondary hover:text-text-primary",
      ghost: "hover:bg-surface-2 text-text-primary",
      danger: "bg-red text-white hover:bg-red/90 shadow-sm",
    };

    const sizes: Record<string, string> = {
      sm: "h-7 px-2.5 text-[0.62rem]",
      md: "h-9 px-4 text-xs",
      lg: "h-11 px-8 text-sm font-semibold uppercase tracking-wider",
      icon: "h-9 w-9",
    };

    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-md font-medium transition-all focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98]",
          variants[variant] || variants.primary,
          sizes[size] || sizes.md,
          className
        )}
        disabled={loading || props.disabled}
        {...props}
      >
        {loading && <Spinner size="sm" className="mr-2 text-current" />}
        {children}
      </button>
    );
  }
);
Button.displayName = "Button";

export function FilterBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
        active
          ? "bg-surface-2 font-medium text-text-primary border border-border/50 shadow-sm"
          : "text-text-muted hover:text-text-secondary"
      }`}
    >
      {children}
    </button>
  );
}

export { Button };
