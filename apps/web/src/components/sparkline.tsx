"use client";

import { useEffect, useRef, useState } from "react";

export function Sparkline({
  runId,
  status,
}: {
  runId: string;
  status: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    fetch(`/api/runs/${encodeURIComponent(runId)}/metrics?last=60`)
      .then((r) => r.json())
      .then((metrics: Array<{ loss: number }>) => {
        if (metrics.length < 2) return;
        const values = metrics.map((m) => m.loss);
        draw(canvas, values, status);
        setLoaded(true);
      })
      .catch(() => {});
  }, [runId, status]);

  return (
    <canvas
      ref={canvasRef}
      className="h-8 w-28 shrink-0 self-center"
      style={{ opacity: loaded ? 1 : 0.3 }}
    />
  );
}

function draw(canvas: HTMLCanvasElement, values: number[], status: string) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const colors: Record<string, string> = {
    completed: "#60a5fa",
    active: "#4ade80",
    stale: "#f59e0b",
    failed: "#f87171",
  };

  ctx.beginPath();
  ctx.strokeStyle = colors[status] ?? "#888";
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";

  for (let i = 0; i < values.length; i++) {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((values[i] - min) / range) * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
