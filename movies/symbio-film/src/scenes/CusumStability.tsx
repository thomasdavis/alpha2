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
  GlowDot,
} from "../components/shared";
import { metrics, sampleMetrics, getRange } from "../data";

// Animated multi-line chart for CUSUM monitors
const CusumChart: React.FC<{ delay: number }> = ({ delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sampled = sampleMetrics(200);
  const width = 1700;
  const height = 280;

  const progress = interpolate(frame - delay, [0, 120], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const cardOpacity = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  const series = [
    {
      label: "Gradient CUSUM",
      color: THEME.red,
      data: sampled.map((m) => Math.min(m.cusum_grad, 50)),
    },
    {
      label: "Clipping CUSUM",
      color: THEME.amber,
      data: sampled.map((m) => Math.min(m.cusum_clip, 50)),
    },
    {
      label: "Throughput CUSUM",
      color: THEME.green,
      data: sampled.map((m) => Math.min(m.cusum_tps, 50)),
    },
    {
      label: "Val Loss CUSUM",
      color: THEME.purple,
      data: sampled.map((m) => Math.min(m.cusum_val, 50)),
    },
  ];

  const maxVal = 50;

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
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <span
          style={{
            fontSize: 18,
            color: THEME.text,
            fontWeight: 600,
            fontFamily,
          }}
        >
          CUSUM Change Detection Monitors
        </span>
        <div style={{ display: "flex", gap: 16 }}>
          {series.map((s) => (
            <div
              key={s.label}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: 12,
                color: THEME.textMuted,
                fontFamily,
              }}
            >
              <GlowDot color={s.color} size={7} />
              {s.label}
            </div>
          ))}
        </div>
      </div>

      <svg width={width} height={height}>
        {/* Threshold line */}
        <line
          x1={0}
          y1={height - (4 / maxVal) * height}
          x2={width}
          y2={height - (4 / maxVal) * height}
          stroke={THEME.red}
          strokeWidth={1}
          strokeDasharray="6,4"
          opacity={0.4}
        />
        <text
          x={width - 10}
          y={height - (4 / maxVal) * height - 6}
          fill={THEME.red}
          fontSize={11}
          textAnchor="end"
          fontFamily={fontFamily}
          opacity={0.6}
        >
          Sensitivity Threshold (4)
        </text>

        {series.map((s) => {
          const pts = s.data.map((v, i) => ({
            x: (i / (s.data.length - 1)) * width,
            y: height - (v / maxVal) * height,
          }));
          const pathD = pts
            .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
            .join(" ");

          let evolved = { strokeDasharray: "none" as string, strokeDashoffset: 0 };
          try {
            evolved = evolvePath(progress, pathD);
          } catch {
            // skip
          }

          return (
            <path
              key={s.label}
              d={pathD}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeDasharray={evolved.strokeDasharray}
              strokeDashoffset={evolved.strokeDashoffset}
              strokeLinecap="round"
              opacity={0.8}
            />
          );
        })}
      </svg>
    </div>
  );
};

// Gradient clipping chart
const ClipChart: React.FC<{ delay: number }> = ({ delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sampled = sampleMetrics(200);
  const width = 800;
  const height = 180;

  const progress = interpolate(frame - delay, [0, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const cardOpacity = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  const clipData = sampled.map((m) => m.clip_coef ?? 1);

  const pts = clipData.map((v, i) => ({
    x: (i / (clipData.length - 1)) * width,
    y: height - v * height,
  }));
  const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  const areaD = pathD + ` L ${width} ${height} L 0 ${height} Z`;

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
          fontSize: 15,
          color: THEME.textMuted,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontWeight: 500,
          marginBottom: 12,
          fontFamily,
        }}
      >
        Gradient Clipping Coefficient
      </div>
      <svg width={width} height={height}>
        <defs>
          <linearGradient id="clipGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={THEME.amber} stopOpacity={0.25} />
            <stop offset="100%" stopColor={THEME.amber} stopOpacity={0} />
          </linearGradient>
        </defs>
        <path d={areaD} fill="url(#clipGrad)" opacity={progress} />
        <path
          d={pathD}
          fill="none"
          stroke={THEME.amber}
          strokeWidth={2}
          opacity={progress}
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
};

// Grad norm chart
const GradNormChart: React.FC<{ delay: number }> = ({ delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const sampled = sampleMetrics(200);
  const width = 800;
  const height = 180;

  const progress = interpolate(frame - delay, [0, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const cardOpacity = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  // Use log scale for grad norm since it has spikes
  const gradData = sampled.map((m) => Math.log10(Math.max(0.001, m.grad_norm)));
  const yMin = Math.min(...gradData);
  const yMax = Math.max(...gradData);
  const range = yMax - yMin || 1;

  const pts = gradData.map((v, i) => ({
    x: (i / (gradData.length - 1)) * width,
    y: height - ((v - yMin) / range) * (height - 10),
  }));
  const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  const areaD = pathD + ` L ${width} ${height} L 0 ${height} Z`;

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
          fontSize: 15,
          color: THEME.textMuted,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontWeight: 500,
          marginBottom: 12,
          fontFamily,
        }}
      >
        Gradient Norm (log scale)
      </div>
      <svg width={width} height={height}>
        <defs>
          <linearGradient id="gnGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={THEME.red} stopOpacity={0.25} />
            <stop offset="100%" stopColor={THEME.red} stopOpacity={0} />
          </linearGradient>
        </defs>
        <path d={areaD} fill="url(#gnGrad)" opacity={progress} />
        <path
          d={pathD}
          fill="none"
          stroke={THEME.red}
          strokeWidth={2}
          opacity={progress}
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
};

// Phase change timeline
const PhaseTimeline: React.FC<{ delay: number }> = ({ delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
    durationInFrames: 40,
  });

  // Count alerts per window
  const windowSize = 50;
  const totalSteps = metrics[metrics.length - 1].step;
  const numWindows = Math.ceil(totalSteps / windowSize);
  const alertDensity: number[] = [];

  for (let w = 0; w < numWindows; w++) {
    const start = w * windowSize;
    const end = start + windowSize;
    const alertsInWindow = metrics.filter(
      (m) => m.step >= start && m.step < end && m.cusum_alerts > 0
    ).length;
    alertDensity.push(alertsInWindow);
  }

  const maxDensity = Math.max(...alertDensity, 1);
  const barWidth = 1700 / numWindows;

  return (
    <div
      style={{
        backgroundColor: THEME.bgCard,
        border: `1px solid ${THEME.border}`,
        borderRadius: 14,
        padding: 24,
        opacity: progress,
      }}
    >
      <div
        style={{
          fontSize: 15,
          color: THEME.textMuted,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontWeight: 500,
          marginBottom: 12,
          fontFamily,
        }}
      >
        CUSUM Alert Density (Phase Changes)
      </div>
      <svg width={1700} height={80}>
        {alertDensity.map((d, i) => {
          const barProgress = spring({
            frame: frame - delay - i * 1,
            fps,
            config: { damping: 18, stiffness: 80 },
          });
          const h = (d / maxDensity) * 60 * barProgress;
          const intensity = d / maxDensity;
          const color = interpolateColor(intensity);
          return (
            <rect
              key={i}
              x={i * barWidth}
              y={60 - h}
              width={Math.max(1, barWidth - 1)}
              height={h}
              fill={color}
              rx={1}
              opacity={0.8}
            />
          );
        })}
      </svg>
    </div>
  );
};

function interpolateColor(t: number): string {
  // Green -> amber -> red
  if (t < 0.5) {
    return THEME.green;
  } else if (t < 0.8) {
    return THEME.amber;
  }
  return THEME.red;
}

export const CusumStabilityScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const totalAlerts = metrics.filter((m) => m.cusum_alerts > 0).length;
  const [gnMin, gnMax] = getRange("grad_norm");

  const fadeOut = interpolate(
    frame,
    [durationInFrames - 30, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SceneBackground>
      <AbsoluteFill style={{ opacity: fadeOut, padding: "50px 80px" }}>
        <SceneTitle
          title="Stability Monitoring"
          subtitle="CUSUM change detection, gradient clipping, and phase analysis"
        />

        {/* Stats */}
        <div style={{ display: "flex", gap: 20, marginBottom: 24 }}>
          <StatCard
            label="CUSUM Alerts"
            value={String(totalAlerts)}
            color={THEME.red}
            delay={20}
          />
          <StatCard
            label="Grad Norm Range"
            value={`${gnMin.toFixed(2)} - ${gnMax.toFixed(0)}`}
            color={THEME.amber}
            delay={25}
          />
          <StatCard
            label="Sensitivity"
            value="4.0"
            color={THEME.accent}
            delay={30}
            sub="CUSUM threshold"
          />
          <StatCard
            label="Grad Clip"
            value="1.0"
            color={THEME.purple}
            delay={35}
            sub="max norm"
          />
        </div>

        {/* CUSUM chart */}
        <CusumChart delay={50} />

        {/* Bottom row */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 20,
            marginTop: 20,
          }}
        >
          <Sequence from={120} layout="none">
            <ClipChart delay={0} />
          </Sequence>
          <Sequence from={140} layout="none">
            <GradNormChart delay={0} />
          </Sequence>
        </div>

        <Sequence from={200} layout="none">
          <div style={{ marginTop: 20 }}>
            <PhaseTimeline delay={0} />
          </div>
        </Sequence>
      </AbsoluteFill>
    </SceneBackground>
  );
};
