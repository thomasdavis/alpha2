"use client";

import { useRef, useMemo, useEffect, useState, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────

interface MetricPoint {
  step: number;
  loss: number;
  val_loss: number | null;
  lr: number;
  grad_norm: number;
  tokens_per_sec: number;
  ms_per_iter: number;
  fitness_score?: number | null;
  symbio_candidate_id?: string | null;
  symbio_candidate_name?: string | null;
  symbio_candidate_activation?: string | null;
  symbio_candidate_parent_id?: string | null;
  symbio_candidate_parent_name?: string | null;
  symbio_generation?: number | null;
  architecture_diversity?: number | null;
  mi_input_repr?: number | null;
  mi_repr_output?: number | null;
}

const ACT_COLORS: Record<string, string> = {
  gelu: "#60a5fa", silu: "#34d399", relu: "#f59e0b", swiglu: "#a78bfa",
  universal: "#f472b6", kan_spline: "#22d3ee", composed: "#e879f9",
  identity: "#94a3b8", square: "#fb923c",
};

/** Get color for an activation name (may be a composed formula). */
function actColor(name: string | null | undefined): string {
  if (!name) return "#888888";
  if (ACT_COLORS[name]) return ACT_COLORS[name];
  // For composed formulas, pick color based on dominant basis
  for (const [key, col] of Object.entries(ACT_COLORS)) {
    if (name.includes(key)) return col;
  }
  return "#e879f9"; // default purple for composed
}

const ACT_SHORT: Record<string, string> = {
  gelu: "GE", silu: "SI", relu: "RE", swiglu: "SW",
  universal: "UN", kan_spline: "KS",
};

// ── Helpers ────────────────────────────────────────────────

function normalizeArr(arr: number[]): number[] {
  if (arr.length === 0) return [];
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = max - min || 1;
  return arr.map((v) => (v - min) / range);
}

function emaSmooth(arr: number[], alpha = 0.08): number[] {
  const out: number[] = [];
  let s = arr[0] ?? 0;
  for (const v of arr) {
    s = alpha * v + (1 - alpha) * s;
    out.push(s);
  }
  return out;
}

function hexToRgba(hex: string, a: number): string {
  if (!hex || hex.length < 7 || hex[0] !== "#") return `rgba(136,136,136,${a})`;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  if (isNaN(r) || isNaN(g) || isNaN(b)) return `rgba(136,136,136,${a})`;
  return `rgba(${r},${g},${b},${a})`;
}

// ── Candidate segment ──────────────────────────────────────

interface CandidateSegment {
  startIdx: number;
  endIdx: number;
  name: string;
  activation: string;
  generation: number;
  parentName: string | null;
  losses: number[];
  bestLoss: number;
  finalLoss: number;
  fitnesses: number[];
}

// ── Canvas draw ────────────────────────────────────────────

function drawActivatorRadial(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  dpr: number,
  metrics: MetricPoint[],
  segments: CandidateSegment[],
  genBoundaries: number[],
  lossSmoothed: number[],
  fitnessSmoothed: number[],
  diversitySmoothed: number[],
  miInputSmoothed: number[],
  miOutputSmoothed: number[],
  hoveredIdx: number | null,
) {
  const cx = w / 2;
  const cy = h / 2;
  const maxR = Math.min(cx, cy) - 50;
  const n = metrics.length;
  if (n < 2) return;

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = "#08080f";
  ctx.fillRect(0, 0, w, h);

  const angleOf = (i: number) => (i / n) * Math.PI * 2 - Math.PI / 2;

  // ─── Layer 1: Outer activation band ───────────────────
  // Thick colored arc segments showing which activation is active
  const bandInner = maxR * 0.88;
  const bandOuter = maxR * 1.0;

  for (const seg of segments) {
    const a1 = angleOf(seg.startIdx);
    const a2 = angleOf(Math.min(seg.endIdx + 1, n));
    const col = actColor(seg.activation);

    // Filled arc segment
    ctx.fillStyle = hexToRgba(col, 0.25);
    ctx.beginPath();
    ctx.arc(cx, cy, bandOuter, a1, a2);
    ctx.arc(cx, cy, bandInner, a2, a1, true);
    ctx.closePath();
    ctx.fill();

    // Inner edge highlight
    ctx.strokeStyle = hexToRgba(col, 0.6);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, bandInner, a1, a2);
    ctx.stroke();

    // Outer edge
    ctx.strokeStyle = hexToRgba(col, 0.3);
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.arc(cx, cy, bandOuter, a1, a2);
    ctx.stroke();

    // Candidate name label (if segment is wide enough)
    const arcLen = (seg.endIdx - seg.startIdx) / n;
    if (arcLen > 0.03) {
      const midA = (a1 + a2) / 2;
      const labelR = (bandInner + bandOuter) / 2;
      const lx = cx + Math.cos(midA) * labelR;
      const ly = cy + Math.sin(midA) * labelR;

      ctx.save();
      ctx.translate(lx, ly);
      // Rotate text to follow the arc
      let textAngle = midA + Math.PI / 2;
      // Flip if on the bottom half so text reads left-to-right
      if (midA > 0 && midA < Math.PI) textAngle += Math.PI;
      ctx.rotate(textAngle);
      ctx.fillStyle = hexToRgba(col, 0.85);
      ctx.font = `bold ${arcLen > 0.06 ? 8 : 7}px monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(seg.name, 0, 0);
      ctx.restore();
    }
  }

  // ─── Layer 2: Generation boundary spokes ──────────────
  ctx.lineWidth = 1;
  for (const gi of genBoundaries) {
    const a = angleOf(gi);
    const gen = metrics[gi]?.symbio_generation ?? 0;
    const hue = (gen * 50 + 200) % 360;
    ctx.strokeStyle = `hsla(${hue}, 60%, 50%, 0.25)`;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * 30, cy + Math.sin(a) * 30);
    ctx.lineTo(cx + Math.cos(a) * bandInner, cy + Math.sin(a) * bandInner);
    ctx.stroke();
    ctx.setLineDash([]);

    // Gen label outside band
    ctx.fillStyle = `hsla(${hue}, 60%, 65%, 0.55)`;
    ctx.font = "bold 9px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const lx = cx + Math.cos(a) * (maxR + 14);
    const ly = cy + Math.sin(a) * (maxR + 14);
    ctx.fillText(`G${gen}`, lx, ly);
  }

  // ─── Subtle radial grid ───────────────────────────────
  ctx.strokeStyle = "rgba(255,255,255,0.03)";
  ctx.lineWidth = 0.5;
  for (let r = 50; r < bandInner; r += 50) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  // ─── Layer 3: Loss ring (activation-colored) ──────────
  const lossBase = maxR * 0.28;
  const lossAmp = maxR * 0.14;

  // Fill with activation colors
  for (const seg of segments) {
    const col = actColor(seg.activation);
    ctx.fillStyle = hexToRgba(col, 0.06);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    for (let i = seg.startIdx; i <= Math.min(seg.endIdx, n - 1); i++) {
      const a = angleOf(i);
      const r = lossBase + lossSmoothed[i] * lossAmp;
      ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
    }
    ctx.closePath();
    ctx.fill();
  }

  // Stroke per segment with activation color
  for (const seg of segments) {
    const col = actColor(seg.activation);
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = seg.startIdx; i <= Math.min(seg.endIdx, n - 1); i++) {
      const a = angleOf(i);
      const r = lossBase + lossSmoothed[i] * lossAmp;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      if (i === seg.startIdx) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // ─── Layer 4: Fitness ring ────────────────────────────
  if (fitnessSmoothed.length === n) {
    const fitBase = maxR * 0.50;
    const fitAmp = maxR * 0.10;

    // Fill
    ctx.fillStyle = "rgba(96,165,250,0.04)";
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const idx = i % n;
      const a = angleOf(i);
      const r = fitBase + fitnessSmoothed[idx] * fitAmp;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();

    // Stroke colored by activation
    for (const seg of segments) {
      const col = actColor(seg.activation);
      ctx.strokeStyle = hexToRgba(col, 0.6);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let i = seg.startIdx; i <= Math.min(seg.endIdx, n - 1); i++) {
        const a = angleOf(i);
        const r = fitBase + fitnessSmoothed[i] * fitAmp;
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r;
        if (i === seg.startIdx) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }

  // ─── Layer 5: Diversity ring ──────────────────────────
  if (diversitySmoothed.length === n) {
    const divBase = maxR * 0.66;
    const divAmp = maxR * 0.08;

    ctx.fillStyle = "rgba(168,139,250,0.04)";
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const idx = i % n;
      const a = angleOf(i);
      const r = divBase + diversitySmoothed[idx] * divAmp;
      if (i === 0) ctx.moveTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
      else ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
    }
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "#a78bfa";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const idx = i % n;
      const a = angleOf(i);
      const r = divBase + diversitySmoothed[idx] * divAmp;
      if (i === 0) ctx.moveTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
      else ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
    }
    ctx.closePath();
    ctx.stroke();
  }

  // ─── Layer 6: MI flow rings (thin) ────────────────────
  if (miInputSmoothed.length === n) {
    const miBase = maxR * 0.78;
    const miAmp = maxR * 0.06;

    // MI input→repr
    ctx.strokeStyle = "rgba(251,191,36,0.5)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const idx = i % n;
      const a = angleOf(i);
      const r = miBase + miInputSmoothed[idx] * miAmp;
      if (i === 0) ctx.moveTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
      else ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
    }
    ctx.closePath();
    ctx.stroke();

    // MI repr→output
    if (miOutputSmoothed.length === n) {
      ctx.strokeStyle = "rgba(52,211,153,0.5)";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      for (let i = 0; i <= n; i++) {
        const idx = i % n;
        const a = angleOf(i);
        const r = miBase + miOutputSmoothed[idx] * miAmp;
        if (i === 0) ctx.moveTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
        else ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }

  // ─── Candidate switch markers ─────────────────────────
  // Small diamond markers at each candidate switch point
  for (let s = 1; s < segments.length; s++) {
    const idx = segments[s].startIdx;
    const a = angleOf(idx);
    const r = lossBase + lossSmoothed[idx] * lossAmp;
    const px = cx + Math.cos(a) * r;
    const py = cy + Math.sin(a) * r;
    const col = actColor(segments[s].activation);

    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.moveTo(px, py - 4);
    ctx.lineTo(px + 3, py);
    ctx.lineTo(px, py + 4);
    ctx.lineTo(px - 3, py);
    ctx.closePath();
    ctx.fill();

    // Thin radial line from center to switch point
    ctx.strokeStyle = hexToRgba(col, 0.15);
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * 25, cy + Math.sin(a) * 25);
    ctx.lineTo(cx + Math.cos(a) * (lossBase - 5), cy + Math.sin(a) * (lossBase - 5));
    ctx.stroke();
  }

  // ─── Center: current candidate info ───────────────────
  const lastM = metrics[n - 1];
  const lastSeg = segments[segments.length - 1];
  const actCol = actColor(lastM?.symbio_candidate_activation);

  // Center glow
  const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, 28);
  gradient.addColorStop(0, hexToRgba(actCol, 0.5));
  gradient.addColorStop(0.6, hexToRgba(actCol, 0.15));
  gradient.addColorStop(1, hexToRgba(actCol, 0));
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(cx, cy, 28, 0, Math.PI * 2);
  ctx.fill();

  // Candidate name
  ctx.fillStyle = actCol;
  ctx.font = "bold 10px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(lastSeg?.name ?? "—", cx, cy - 8);

  // Activation type
  ctx.fillStyle = hexToRgba(actCol, 0.7);
  ctx.font = "8px monospace";
  ctx.fillText(lastM?.symbio_candidate_activation ?? "—", cx, cy + 3);

  // Loss value
  ctx.fillStyle = "rgba(255,255,255,0.4)";
  ctx.font = "8px monospace";
  ctx.fillText(lastM ? lastM.loss.toFixed(3) : "—", cx, cy + 13);

  // ─── Hover interaction ────────────────────────────────
  if (hoveredIdx !== null && hoveredIdx >= 0 && hoveredIdx < n) {
    const m = metrics[hoveredIdx];
    const a = angleOf(hoveredIdx);

    // Crosshair line
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a) * (bandOuter + 5), cy + Math.sin(a) * (bandOuter + 5));
    ctx.stroke();
    ctx.setLineDash([]);

    // Highlight dot on loss ring
    const lr = lossBase + lossSmoothed[hoveredIdx] * lossAmp;
    const hx = cx + Math.cos(a) * lr;
    const hy = cy + Math.sin(a) * lr;
    const hCol = actColor(m.symbio_candidate_activation);
    ctx.fillStyle = hCol;
    ctx.beginPath();
    ctx.arc(hx, hy, 3, 0, Math.PI * 2);
    ctx.fill();

    // Find which segment this belongs to
    const seg = segments.find((s) => hoveredIdx >= s.startIdx && hoveredIdx <= s.endIdx);

    // Tooltip
    const tx = cx + Math.cos(a) * (maxR * 0.55);
    const ty = cy + Math.sin(a) * (maxR * 0.55);

    const act = m.symbio_candidate_activation || "—";
    const gen = m.symbio_generation ?? "—";
    const lines = [
      `${seg?.name ?? "—"} (${act})`,
      `step ${m.step} · gen ${gen}`,
      `loss ${m.loss.toFixed(4)}`,
    ];
    if (m.fitness_score != null) lines.push(`fitness ${m.fitness_score.toFixed(4)}`);
    if (m.architecture_diversity != null) lines.push(`diversity ${m.architecture_diversity.toFixed(3)}`);
    if (m.mi_input_repr != null) lines.push(`MI in→h ${m.mi_input_repr.toFixed(3)}`);
    if (m.mi_repr_output != null) lines.push(`MI h→out ${m.mi_repr_output.toFixed(3)}`);
    if (seg?.parentName) lines.push(`parent: ${seg.parentName}`);

    const boxW = 130;
    const boxH = lines.length * 13 + 10;
    const bx = Math.min(Math.max(tx - boxW / 2, 5), w - boxW - 5);
    const by = Math.min(Math.max(ty - boxH - 8, 5), h - boxH - 5);

    ctx.fillStyle = "rgba(0,0,0,0.88)";
    ctx.strokeStyle = hexToRgba(hCol, 0.3);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(bx, by, boxW, boxH, 4);
    ctx.fill();
    ctx.stroke();

    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    lines.forEach((line, li) => {
      ctx.fillStyle = li === 0 ? hCol : "rgba(255,255,255,0.75)";
      ctx.fillText(line, bx + 6, by + 13 + li * 13);
    });
  }

  // ─── Ring labels ──────────────────────────────────────
  ctx.font = "bold 7px monospace";
  ctx.textAlign = "left";
  const labelAngle = -Math.PI / 2 - 0.18;

  const ringLabels: [string, string, number][] = [
    ["loss", "#60a5fa", lossBase + lossAmp * 0.5],
  ];
  if (fitnessSmoothed.length === n) ringLabels.push(["fitness", "#60a5fa", maxR * 0.50 + maxR * 0.05]);
  if (diversitySmoothed.length === n) ringLabels.push(["diversity", "#a78bfa", maxR * 0.66 + maxR * 0.04]);
  if (miInputSmoothed.length === n) ringLabels.push(["MI flow", "#fbbf24", maxR * 0.78 + maxR * 0.03]);

  for (const [label, color, r] of ringLabels) {
    const lx = cx + Math.cos(labelAngle) * r + 4;
    const ly = cy + Math.sin(labelAngle) * r;
    ctx.fillStyle = hexToRgba(color, 0.5);
    ctx.fillText(label, lx, ly);
  }

  // Step labels around the outside (sparse — 8 positions)
  ctx.fillStyle = "rgba(255,255,255,0.15)";
  ctx.font = "8px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const totalSteps = metrics[n - 1].step;
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2 - Math.PI / 2;
    const stepLabel = Math.round((i / 8) * totalSteps);
    const lx = cx + Math.cos(a) * (maxR + 30);
    const ly = cy + Math.sin(a) * (maxR + 30);
    ctx.fillText(String(stepLabel), lx, ly);
  }

  ctx.restore();
}

// ── Component ──────────────────────────────────────────────

export function RadialTrainingViz({ metrics }: { metrics: MetricPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const n = metrics.length;

  const computed = useMemo(() => {
    if (n < 2) return null;

    // Build candidate segments
    const segments: CandidateSegment[] = [];
    let segStart = 0;
    let prevId = metrics[0]?.symbio_candidate_id ?? null;
    let segLosses: number[] = [metrics[0]?.loss ?? 0];
    let segFitnesses: number[] = [];
    if (metrics[0]?.fitness_score != null) segFitnesses.push(metrics[0].fitness_score);

    for (let i = 1; i < n; i++) {
      const m = metrics[i];
      const id = m.symbio_candidate_id ?? null;
      if (id !== prevId || i === n - 1) {
        const endIdx = i === n - 1 && id === prevId ? i : i - 1;
        if (i === n - 1 && id === prevId) {
          segLosses.push(m.loss);
          if (m.fitness_score != null) segFitnesses.push(m.fitness_score);
        }
        const refM = metrics[segStart];
        segments.push({
          startIdx: segStart,
          endIdx,
          name: refM.symbio_candidate_name ?? prevId ?? "unknown",
          activation: refM.symbio_candidate_activation ?? "?",
          generation: refM.symbio_generation ?? 0,
          parentName: refM.symbio_candidate_parent_name ?? null,
          losses: segLosses,
          bestLoss: segLosses.length > 0 ? Math.min(...segLosses) : 0,
          finalLoss: segLosses[segLosses.length - 1] ?? 0,
          fitnesses: segFitnesses,
        });
        // Start new segment
        if (i < n - 1 || id !== prevId) {
          segStart = i;
          prevId = id;
          segLosses = [m.loss];
          segFitnesses = m.fitness_score != null ? [m.fitness_score] : [];
          // Handle last metric when it's a new segment by itself
          if (i === n - 1) {
            segments.push({
              startIdx: i,
              endIdx: i,
              name: m.symbio_candidate_name ?? id ?? "unknown",
              activation: m.symbio_candidate_activation ?? "?",
              generation: m.symbio_generation ?? 0,
              parentName: m.symbio_candidate_parent_name ?? null,
              losses: segLosses,
              bestLoss: m.loss,
              finalLoss: m.loss,
              fitnesses: segFitnesses,
            });
          }
        }
      } else {
        segLosses.push(m.loss);
        if (m.fitness_score != null) segFitnesses.push(m.fitness_score);
      }
    }

    // Generation boundaries
    const genB: number[] = [];
    let prevGen = -1;
    for (let i = 0; i < n; i++) {
      const g = metrics[i].symbio_generation ?? -1;
      if (g !== prevGen && g >= 0) {
        genB.push(i);
        prevGen = g;
      }
    }

    // Normalized + smoothed series
    const lossRaw = metrics.map((m) => m.loss);
    const lossN = normalizeArr(lossRaw);
    const lossSmoothed = emaSmooth(lossN, 0.06);

    const hasFitness = metrics.some((m) => m.fitness_score != null);
    const fitnessSmoothed = hasFitness
      ? emaSmooth(normalizeArr(metrics.map((m) => m.fitness_score ?? 0)), 0.06)
      : [];

    const hasDiversity = metrics.some((m) => m.architecture_diversity != null);
    const diversitySmoothed = hasDiversity
      ? emaSmooth(normalizeArr(metrics.map((m) => m.architecture_diversity ?? 0)), 0.06)
      : [];

    const hasMI = metrics.some((m) => m.mi_input_repr != null);
    const miInputSmoothed = hasMI
      ? emaSmooth(normalizeArr(metrics.map((m) => m.mi_input_repr ?? 0)), 0.06)
      : [];
    const miOutputSmoothed = hasMI
      ? emaSmooth(normalizeArr(metrics.map((m) => m.mi_repr_output ?? 0)), 0.06)
      : [];

    return { segments, genBoundaries: genB, lossSmoothed, fitnessSmoothed, diversitySmoothed, miInputSmoothed, miOutputSmoothed };
  }, [metrics, n]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !computed || n < 2) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width;
    const h = rect.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawActivatorRadial(
      ctx, w, h, dpr, metrics, computed.segments, computed.genBoundaries,
      computed.lossSmoothed, computed.fitnessSmoothed, computed.diversitySmoothed,
      computed.miInputSmoothed, computed.miOutputSmoothed, hoveredIdx,
    );
  }, [computed, metrics, hoveredIdx, n]);

  useEffect(() => { draw(); }, [draw]);

  useEffect(() => {
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  // Mouse interaction
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const cxp = rect.width / 2;
    const cyp = rect.height / 2;
    const dx = mx - cxp;
    const dy = my - cyp;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 15 || dist > Math.min(cxp, cyp) - 10) {
      setHoveredIdx(null);
    } else {
      let angle = Math.atan2(dy, dx);
      // Convert from canvas angle to index
      let norm = (angle + Math.PI / 2) / (Math.PI * 2);
      if (norm < 0) norm += 1;
      const idx = Math.round(norm * n) % n;
      setHoveredIdx(Math.max(0, Math.min(n - 1, idx)));
    }
  }, [n]);

  const handleMouseLeave = useCallback(() => setHoveredIdx(null), []);

  // Don't render if no symbio data
  const hasSymbioData = metrics.some((m) => m.symbio_candidate_id != null);
  if (n < 5 || !hasSymbioData) return null;

  // Build activation summary for legend
  const actCounts = new Map<string, number>();
  for (const m of metrics) {
    const act = m.symbio_candidate_activation;
    if (act) actCounts.set(act, (actCounts.get(act) ?? 0) + 1);
  }
  const actList = Array.from(actCounts.entries()).sort((a, b) => b[1] - a[1]);

  return (
    <div className="relative w-full overflow-hidden rounded-lg border border-border bg-[#08080f]">
      {/* Activation legend with step counts */}
      <div className="absolute top-2 right-2 z-10 flex flex-wrap gap-1 pointer-events-none">
        {actList.map(([name, count]) => (
          <span key={name} className="flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-[0.55rem] backdrop-blur-sm" style={{ color: actColor(name) }}>
            <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ backgroundColor: actColor(name) }} />
            {name}
            <span className="text-white/30">{count}</span>
          </span>
        ))}
      </div>

      {/* Info */}
      <div className="absolute bottom-2 left-2 z-10 pointer-events-none">
        <span className="rounded bg-black/60 px-1.5 py-0.5 font-mono text-[0.6rem] text-text-muted/60 backdrop-blur-sm">
          {computed?.segments.length ?? 0} candidates · {computed?.genBoundaries.length ?? 0} generations · angle = time
        </span>
      </div>

      <canvas
        ref={canvasRef}
        className="h-[460px] w-full cursor-crosshair"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  );
}
