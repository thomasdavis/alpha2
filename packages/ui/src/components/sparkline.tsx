"use client";

import * as React from "react";

export interface SparklineProps {
  data: number[];
  variant?: "success" | "blue" | "warning" | "danger" | "default";
  className?: string;
}

export function Sparkline({
  data,
  variant = "default",
  className = "h-8 w-28",
}: SparklineProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Get color from CSS variables defined in globals.css
    const style = getComputedStyle(document.documentElement);
    const colors = {
      success: style.getPropertyValue("--green").trim() || "#4ade80",
      blue: style.getPropertyValue("--blue").trim() || "#60a5fa",
      warning: style.getPropertyValue("--yellow").trim() || "#f59e0b",
      danger: style.getPropertyValue("--red").trim() || "#f87171",
      default: style.getPropertyValue("--text-muted").trim() || "#888",
    };
    
    const color = colors[variant];

    if (data.length === 1) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, 3, 0, Math.PI * 2);
      ctx.fill();
      return;
    }

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";

    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((data[i] - min) / range) * (h - 4) - 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [data, variant]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
    />
  );
}
