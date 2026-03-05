import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Easing,
  Sequence,
} from "remotion";
import { evolvePath } from "@remotion/paths";
import { THEME } from "../theme";
import {
  fontFamily,
  SceneBackground,
  SceneTitle,
  FadeIn,
  StatCard,
} from "../components/shared";
import { metrics, sampleMetrics, getRange, run } from "../data";

// Mini chart component for this scene
const MiniLineChart: React.FC<{
  data: number[];
  color: string;
  width: number;
  height: number;
  delay: number;
  label: string;
  suffix?: string;
  yMin?: number;
  yMax?: number;
}> = ({ data, color, width, height, delay, label, suffix = "", yMin: forcedMin, yMax: forcedMax }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const yMin = forcedMin ?? Math.min(...data);
  const yMax = forcedMax ?? Math.max(...data);
  const range = yMax - yMin || 1;

  const progress = interpolate(frame - delay, [0, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const cardOpacity = spring({
    frame: frame - delay + 10,
    fps,
    config: { damping: 200 },
  });

  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - yMin) / range) * (height - 20);
    return { x, y };
  });

  const pathD = pts
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
    .join(" ");

  const areaD =
    pathD + ` L ${width} ${height} L 0 ${height} Z`;

  let evolved = { strokeDasharray: "none", strokeDashoffset: 0 };
  try {
    evolved = evolvePath(progress, pathD);
  } catch {
    // path too short
  }

  const currentIdx = Math.min(
    data.length - 1,
    Math.floor(progress * data.length)
  );
  const currentVal = data[currentIdx];

  return (
    <div
      style={{
        backgroundColor: THEME.bgCard,
        border: `1px solid ${THEME.border}`,
        borderRadius: 14,
        padding: 24,
        opacity: cardOpacity,
        transform: `translateY(${(1 - cardOpacity) * 20}px)`,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 16,
        }}
      >
        <span
          style={{
            fontSize: 15,
            color: THEME.textMuted,
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            fontWeight: 500,
            fontFamily,
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontSize: 24,
            fontWeight: 700,
            color,
            fontFeatureSettings: '"tnum"',
            fontFamily,
          }}
        >
          {typeof currentVal === "number" ? currentVal.toFixed(0) : "--"}
          {suffix}
        </span>
      </div>

      <svg width={width} height={height}>
        <defs>
          <linearGradient
            id={`grad-${label.replace(/\s/g, "")}`}
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop offset="0%" stopColor={color} stopOpacity={0.2} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <path
          d={areaD}
          fill={`url(#grad-${label.replace(/\s/g, "")})`}
          opacity={progress}
        />
        <path
          d={pathD}
          fill="none"
          stroke={color}
          strokeWidth={2.5}
          strokeDasharray={evolved.strokeDasharray}
          strokeDashoffset={evolved.strokeDashoffset}
          strokeLinecap="round"
        />
        {progress > 0.05 && (
          <circle
            cx={pts[currentIdx].x}
            cy={pts[currentIdx].y}
            r={5}
            fill={color}
            style={{ filter: `drop-shadow(0 0 8px ${color}80)` }}
          />
        )}
      </svg>
    </div>
  );
};

// Stacked timing bar chart
const TimingBreakdown: React.FC<{ delay: number }> = ({ delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sampled = sampleMetrics(30);

  const phases = [
    { key: "timing_bwd_ms" as const, label: "Backward", color: THEME.red },
    { key: "timing_fwd_ms" as const, label: "Forward", color: THEME.accent },
    { key: "timing_grad_norm_ms" as const, label: "Grad Norm", color: THEME.purple },
    { key: "timing_optim_ms" as const, label: "Optimizer", color: THEME.green },
    { key: "timing_grad_clip_ms" as const, label: "Grad Clip", color: THEME.amber },
    { key: "timing_data_ms" as const, label: "Data", color: THEME.pink },
  ];

  const barWidth = 38;
  const barGap = 8;
  const chartHeight = 250;
  const totalWidth = sampled.length * (barWidth + barGap);

  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
    durationInFrames: 60,
  });

  return (
    <div
      style={{
        backgroundColor: THEME.bgCard,
        border: `1px solid ${THEME.border}`,
        borderRadius: 14,
        padding: 24,
        opacity: progress,
        transform: `translateY(${(1 - progress) * 30}px)`,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <span
          style={{
            fontSize: 15,
            color: THEME.textMuted,
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            fontWeight: 500,
            fontFamily,
          }}
        >
          Step Time Breakdown
        </span>
        <div style={{ display: "flex", gap: 14 }}>
          {phases.map((p) => (
            <div
              key={p.label}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 5,
                fontSize: 11,
                color: THEME.textMuted,
                fontFamily,
              }}
            >
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 2,
                  backgroundColor: p.color,
                }}
              />
              {p.label}
            </div>
          ))}
        </div>
      </div>

      <svg width={totalWidth} height={chartHeight + 30} style={{ overflow: "visible" }}>
        {sampled.map((m, i) => {
          const total =
            m.timing_fwd_ms +
            m.timing_bwd_ms +
            m.timing_grad_norm_ms +
            m.timing_optim_ms +
            m.timing_grad_clip_ms +
            m.timing_data_ms;

          const barProgress = spring({
            frame: frame - delay - i * 2,
            fps,
            config: { damping: 18, stiffness: 80 },
          });

          let yOffset = 0;
          return (
            <g key={i}>
              {phases.map((phase) => {
                const val = m[phase.key] as number;
                const h = (val / total) * chartHeight * barProgress;
                const y = chartHeight - yOffset - h;
                yOffset += h;
                return (
                  <rect
                    key={phase.key}
                    x={i * (barWidth + barGap)}
                    y={y}
                    width={barWidth}
                    height={Math.max(0, h)}
                    fill={phase.color}
                    rx={2}
                    opacity={0.85}
                  />
                );
              })}
              {i % 5 === 0 && (
                <text
                  x={i * (barWidth + barGap) + barWidth / 2}
                  y={chartHeight + 20}
                  fill={THEME.textDim}
                  fontSize={10}
                  textAnchor="middle"
                  fontFamily={fontFamily}
                >
                  {m.step}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
};

export const GpuInfraScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const sampled = sampleMetrics(100);
  const vramData = sampled.map((m) => m.gpu_vram_used_mb);
  const gpuUtilData = sampled.map((m) => m.gpu_util_pct);
  const tpsData = sampled.map((m) => m.tokens_per_sec);
  const iterData = sampled.map((m) => m.ms_per_iter);

  const fadeOut = interpolate(
    frame,
    [durationInFrames - 30, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SceneBackground>
      <AbsoluteFill style={{ opacity: fadeOut, padding: "60px 80px" }}>
        <SceneTitle
          title="GPU & Infrastructure"
          subtitle={`${run.gpu_name} | ${run.hostname} | ${run.cpu_count} CPU | ${(run.ram_total_mb / 1024).toFixed(0)}GB RAM`}
        />

        {/* Stats row */}
        <div
          style={{
            display: "flex",
            gap: 20,
            marginBottom: 30,
          }}
        >
          <StatCard label="GPU" value={run.gpu_name || ""} color={THEME.green} delay={20} />
          <StatCard
            label="VRAM"
            value={`${(run.gpu_vram_mb / 1024).toFixed(0)}GB`}
            color={THEME.amber}
            delay={25}
          />
          <StatCard
            label="Peak TPS"
            value={Math.max(...tpsData).toFixed(0)}
            color={THEME.accent}
            delay={30}
            sub="tokens/sec"
          />
          <StatCard
            label="Avg Step"
            value={`${(iterData.reduce((a, b) => a + b, 0) / iterData.length / 1000).toFixed(1)}s`}
            color={THEME.purple}
            delay={35}
          />
        </div>

        {/* Charts grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 20,
            marginBottom: 20,
          }}
        >
          <MiniLineChart
            data={vramData}
            color={THEME.green}
            width={780}
            height={160}
            delay={50}
            label="VRAM Usage"
            suffix=" MB"
          />
          <MiniLineChart
            data={gpuUtilData}
            color={THEME.amber}
            width={780}
            height={160}
            delay={60}
            label="GPU Utilization"
            suffix="%"
            yMin={0}
            yMax={100}
          />
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 20,
            marginBottom: 20,
          }}
        >
          <MiniLineChart
            data={tpsData}
            color={THEME.accent}
            width={780}
            height={160}
            delay={70}
            label="Tokens / Second"
          />
          <MiniLineChart
            data={iterData.map((v) => v / 1000)}
            color={THEME.purple}
            width={780}
            height={160}
            delay={80}
            label="Step Time"
            suffix="s"
          />
        </div>

        <Sequence from={100} layout="none">
          <TimingBreakdown delay={0} />
        </Sequence>
      </AbsoluteFill>
    </SceneBackground>
  );
};
