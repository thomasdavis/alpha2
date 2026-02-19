"use client";

import { useEffect, useRef, useState } from "react";

interface Metric {
  step: number;
  loss: number;
  val_loss: number | null;
}

export function LossChart({ runId }: { runId: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/runs/${encodeURIComponent(runId)}/metrics`)
      .then((r) => r.json())
      .then((metrics: Metric[]) => {
        const canvas = canvasRef.current;
        if (canvas && metrics.length >= 2) draw(canvas, metrics);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [runId]);

  return (
    <div className="relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-text-muted">
          Loading chart...
        </div>
      )}
      <canvas ref={canvasRef} className="h-52 w-full" />
    </div>
  );
}

function draw(canvas: HTMLCanvasElement, metrics: Metric[]) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { top: 10, right: 12, bottom: 28, left: 50 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const losses = metrics.map((m) => m.loss);
  const valPts = metrics.filter((m) => m.val_loss != null) as Array<{
    step: number;
    val_loss: number;
  }>;
  const allVals = [...losses, ...valPts.map((v) => v.val_loss)];
  const minL = Math.min(...allVals);
  const maxL = Math.max(...allVals);
  const rangeL = maxL - minL || 1;
  const maxStep = metrics[metrics.length - 1].step;
  const minStep = metrics[0].step;
  const rangeS = maxStep - minStep || 1;

  const sx = (step: number) => pad.left + ((step - minStep) / rangeS) * cw;
  const sy = (loss: number) => pad.top + (1 - (loss - minL) / rangeL) * ch;

  // Grid lines
  ctx.strokeStyle = "#222";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    const val = maxL - (i / 4) * rangeL;
    ctx.fillStyle = "#555";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.fillText(val.toFixed(2), pad.left - 6, y + 3);
  }

  // Step labels
  ctx.textAlign = "center";
  const ticks = [
    minStep,
    Math.round(minStep + rangeS * 0.25),
    Math.round(minStep + rangeS * 0.5),
    Math.round(minStep + rangeS * 0.75),
    maxStep,
  ];
  for (const s of ticks) {
    ctx.fillStyle = "#555";
    ctx.font = "10px monospace";
    ctx.fillText(s.toString(), sx(s), h - pad.bottom + 14);
  }

  // Train loss line
  ctx.beginPath();
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  for (let i = 0; i < metrics.length; i++) {
    const x = sx(metrics[i].step);
    const y = sy(metrics[i].loss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Val loss dots + line
  if (valPts.length > 0) {
    ctx.beginPath();
    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 1;
    for (let i = 0; i < valPts.length; i++) {
      const x = sx(valPts[i].step);
      const y = sy(valPts[i].val_loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = "#60a5fa";
    for (const v of valPts) {
      ctx.beginPath();
      ctx.arc(sx(v.step), sy(v.val_loss), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Legend
  const ly = h - 6;
  ctx.fillStyle = "#f59e0b";
  ctx.fillRect(pad.left, ly, 12, 2);
  ctx.fillStyle = "#888";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("train loss", pad.left + 16, ly + 3);
  if (valPts.length > 0) {
    ctx.fillStyle = "#60a5fa";
    ctx.beginPath();
    ctx.arc(pad.left + 90, ly + 1, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#888";
    ctx.fillText("val loss", pad.left + 97, ly + 3);
  }
}
