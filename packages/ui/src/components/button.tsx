import * as React from "react";
import { cn } from "../utils.js";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
  size?: "sm" | "md" | "lg" | "icon";
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", ...props }, ref) => {
    const variants: Record<string, string> = {
      primary: "bg-accent text-white hover:bg-accent/90",
      secondary: "bg-surface-2 text-text-primary hover:bg-surface-2/80 border border-border",
      outline: "border border-border bg-transparent hover:bg-surface-2 text-text-secondary hover:text-text-primary",
      ghost: "hover:bg-surface-2 text-text-primary",
      danger: "bg-red text-white hover:bg-red/90",
    };

    const sizes: Record<string, string> = {
      sm: "h-8 px-3 text-[0.65rem]",
      md: "h-9 px-4 text-xs",
      lg: "h-10 px-6 text-sm",
      icon: "h-9 w-9",
    };

    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent disabled:pointer-events-none disabled:opacity-50",
          variants[variant] || variants.primary,
          sizes[size] || sizes.md,
          className
        )}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button };
