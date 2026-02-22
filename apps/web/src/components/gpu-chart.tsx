"use client";

import { useEffect, useRef, useState } from "react";

interface GpuMetric {
  step: number;
  gpu_util_pct: number | null;
  gpu_vram_used_mb: number | null;
  gpu_vram_total_mb: number | null;
  gpu_mem_pool_mb: number | null;
}

export function GpuChart({ runId }: { runId: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);
  const [hasData, setHasData] = useState(false);

  useEffect(() => {
    fetch(`/api/runs/${encodeURIComponent(runId)}/metrics`)
      .then((r) => r.json())
      .then((metrics: GpuMetric[]) => {
        const gpuMetrics = metrics.filter(
          (m) => m.gpu_vram_used_mb != null || m.gpu_util_pct != null
        );
        const canvas = canvasRef.current;
        if (canvas && gpuMetrics.length >= 1) {
          draw(canvas, gpuMetrics);
          setHasData(true);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [runId]);

  if (!loading && !hasData) return null;

  return (
    <div className="relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-text-muted">
          Loading GPU data...
        </div>
      )}
      <canvas ref={canvasRef} className="h-52 w-full" />
    </div>
  );
}

function draw(canvas: HTMLCanvasElement, metrics: GpuMetric[]) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { top: 10, right: 50, bottom: 28, left: 50 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const maxStep = metrics[metrics.length - 1].step;
  const minStep = metrics[0].step;
  const rangeS = maxStep - minStep || 1;

  const sx = (step: number) => pad.left + ((step - minStep) / rangeS) * cw;

  // Left axis: VRAM (MB)
  const vramPts = metrics.filter((m) => m.gpu_vram_used_mb != null);
  const totalMb = vramPts[0]?.gpu_vram_total_mb ?? 24000;
  const maxVram = totalMb;
  const syVram = (mb: number) => pad.top + (1 - mb / maxVram) * ch;

  // Right axis: Utilization (0-100%)
  const syUtil = (pct: number) => pad.top + (1 - pct / 100) * ch;

  // Grid lines
  ctx.strokeStyle = "#222";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    // Left axis labels (VRAM)
    const vramVal = maxVram * (1 - i / 4);
    ctx.fillStyle = "#10b981";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.fillText((vramVal / 1024).toFixed(1) + "G", pad.left - 6, y + 3);

    // Right axis labels (util %)
    const utilVal = 100 * (1 - i / 4);
    ctx.fillStyle = "#f59e0b";
    ctx.textAlign = "left";
    ctx.fillText(utilVal.toFixed(0) + "%", w - pad.right + 6, y + 3);
  }

  // Step labels
  ctx.textAlign = "center";
  ctx.fillStyle = "#555";
  ctx.font = "10px monospace";
  if (metrics.length === 1) {
    ctx.fillText(minStep.toString(), sx(minStep), h - pad.bottom + 14);
  } else {
    const ticks = [
      minStep,
      Math.round(minStep + rangeS * 0.25),
      Math.round(minStep + rangeS * 0.5),
      Math.round(minStep + rangeS * 0.75),
      maxStep,
    ];
    for (const s of ticks) {
      ctx.fillText(s.toString(), sx(s), h - pad.bottom + 14);
    }
  }

  // VRAM usage line (green)
  if (vramPts.length > 0) {
    ctx.beginPath();
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    for (let i = 0; i < vramPts.length; i++) {
      const x = sx(vramPts[i].step);
      const y = syVram(vramPts[i].gpu_vram_used_mb!);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // GPU utilization line (amber)
  const utilPts = metrics.filter((m) => m.gpu_util_pct != null);
  if (utilPts.length > 0) {
    ctx.beginPath();
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    for (let i = 0; i < utilPts.length; i++) {
      const x = sx(utilPts[i].step);
      const y = syUtil(utilPts[i].gpu_util_pct!);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Pool memory line (purple, dashed)
  const poolPts = metrics.filter((m) => m.gpu_mem_pool_mb != null);
  if (poolPts.length > 0) {
    ctx.beginPath();
    ctx.strokeStyle = "#a78bfa";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.lineJoin = "round";
    for (let i = 0; i < poolPts.length; i++) {
      const x = sx(poolPts[i].step);
      const y = syVram(poolPts[i].gpu_mem_pool_mb!);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Legend
  const ly = h - 6;
  ctx.fillStyle = "#10b981";
  ctx.fillRect(pad.left, ly, 12, 2);
  ctx.fillStyle = "#888";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("VRAM", pad.left + 16, ly + 3);

  ctx.fillStyle = "#f59e0b";
  ctx.fillRect(pad.left + 62, ly, 12, 2);
  ctx.fillStyle = "#888";
  ctx.fillText("GPU %", pad.left + 78, ly + 3);

  if (poolPts.length > 0) {
    ctx.strokeStyle = "#a78bfa";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(pad.left + 130, ly + 1);
    ctx.lineTo(pad.left + 142, ly + 1);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#888";
    ctx.fillText("Pool", pad.left + 146, ly + 3);
  }
}
