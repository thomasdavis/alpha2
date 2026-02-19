// ASCII line chart renderer — no external deps, pure string output

const BLOCKS = "▁▂▃▄▅▆▇█";

// ── Sparkline with per-character color info ───────────────────────────────

export interface SparkChar {
  char: string;
  intensity: number; // 0 (low/good) to 1 (high/bad)
}

export function sparkline(values: number[], width: number): SparkChar[] {
  if (values.length === 0) return [];
  const sampled = sample(values, width);
  const min = Math.min(...sampled);
  const max = Math.max(...sampled);
  const range = max - min || 1;
  return sampled.map(v => {
    const norm = (v - min) / range;
    const idx = Math.round(norm * (BLOCKS.length - 1));
    return { char: BLOCKS[idx], intensity: norm };
  });
}

export function sparklineString(values: number[], width: number): string {
  return sparkline(values, width).map(s => s.char).join("");
}

function sample(values: number[], width: number): number[] {
  if (values.length <= width) return values;
  const result: number[] = [];
  for (let i = 0; i < width; i++) {
    const idx = Math.floor((i / width) * values.length);
    result.push(values[idx]);
  }
  return result;
}

// ── Full ASCII chart ──────────────────────────────────────────────────────

export interface ChartSeries {
  values: number[];
  label: string;
  char: string;
}

export function renderChart(
  series: ChartSeries[],
  width: number,
  height: number,
): string[] {
  if (series.length === 0 || series.every(s => s.values.length === 0)) {
    return [" ".repeat(width)];
  }

  const allValues = series.flatMap(s => s.values).filter(v => v != null);
  if (allValues.length === 0) return [" ".repeat(width)];

  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const range = max - min || 1;

  const labelW = 8;
  const plotW = width - labelW - 1;
  const plotH = height;

  const grid: string[][] = [];
  for (let r = 0; r < plotH; r++) {
    grid.push(new Array(plotW).fill(" "));
  }

  // Plot each series — connect points with vertical fills for smoother look
  for (const s of series) {
    const sampled = sample(s.values, plotW);
    let prevRow = -1;
    for (let x = 0; x < sampled.length; x++) {
      const v = sampled[x];
      if (v == null) continue;
      const row = plotH - 1 - Math.round(((v - min) / range) * (plotH - 1));
      const r = Math.max(0, Math.min(plotH - 1, row));
      grid[r][x] = s.char;

      // Connect to previous point
      if (prevRow >= 0 && Math.abs(r - prevRow) > 1) {
        const from = Math.min(r, prevRow) + 1;
        const to = Math.max(r, prevRow);
        for (let fillR = from; fillR < to; fillR++) {
          if (grid[fillR][x] === " ") grid[fillR][x] = "│";
        }
      }
      prevRow = r;
    }
  }

  const lines: string[] = [];
  for (let r = 0; r < plotH; r++) {
    const yVal = max - (r / (plotH - 1)) * range;
    const label = yVal.toFixed(2).padStart(labelW - 1);
    const border = r === 0 ? "┐" : r === plotH - 1 ? "┤" : "│";
    lines.push(`${label} ${border}${grid[r].join("")}`);
  }

  const axisLine = " ".repeat(labelW - 1) + " └" + "─".repeat(plotW);
  lines.push(axisLine);

  const maxSteps = Math.max(...series.map(s => s.values.length));
  const xStart = " step 0";
  const xEnd = String(maxSteps);
  const pad = plotW + 1 - xStart.length - xEnd.length;
  lines.push(" ".repeat(labelW) + xStart + (pad > 0 ? " ".repeat(pad) : " ") + xEnd);

  const legend = series.map(s => `${s.char} ${s.label}`).join("   ");
  lines.push(" ".repeat(labelW) + legend);

  return lines;
}
