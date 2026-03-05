import * as React from "react";
import { cn } from "../utils.js";
import { Card, CardContent } from "./card.js";

export function EmptyState({ 
  icon, 
  title, 
  description, 
  action, 
  className 
}: { 
  icon?: React.ReactNode; 
  title: string; 
  description?: string; 
  action?: React.ReactNode; 
  className?: string; 
}) {
  return (
    <Card className={cn("py-12 text-center border-dashed border-border/50", className)}>
      <CardContent className="flex flex-col items-center justify-center space-y-4">
        {icon && (
          <div className="flex h-12 w-12 items-center justify-center rounded-full border border-border bg-surface-2 text-text-muted">
            {icon}
          </div>
        )}
        <div className="space-y-1">
          <h3 className="text-sm font-semibold text-text-primary">{title}</h3>
          {description && <p className="text-xs text-text-muted max-w-sm mx-auto">{description}</p>}
        </div>
        {action && <div>{action}</div>}
      </CardContent>
    </Card>
  );
}
