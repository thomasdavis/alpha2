"use client";

import { useState, useRef, useEffect, type ReactNode } from "react";

interface TooltipProps {
  text: string;
  children: ReactNode;
}

export function Tooltip({ text, children }: TooltipProps) {
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState<"top" | "bottom">("top");
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (show && ref.current) {
      const rect = ref.current.getBoundingClientRect();
      setPos(rect.top < 80 ? "bottom" : "top");
    }
  }, [show]);

  return (
    <span
      ref={ref}
      className="relative inline-flex cursor-help"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <span
          className={`absolute z-50 w-56 rounded-lg border border-border-2 bg-surface-2 px-3 py-2 text-xs leading-relaxed text-text-primary shadow-lg ${
            pos === "top"
              ? "bottom-full left-1/2 mb-2 -translate-x-1/2"
              : "top-full left-1/2 mt-2 -translate-x-1/2"
          }`}
        >
          {text}
        </span>
      )}
    </span>
  );
}

/** Small "?" icon that shows a tooltip on hover */
export function Tip({ text }: { text: string }) {
  return (
    <Tooltip text={text}>
      <span className="ml-1 inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-border-2 text-[0.55rem] font-bold text-text-muted transition-colors hover:border-text-muted hover:text-text-secondary">
        ?
      </span>
    </Tooltip>
  );
}
