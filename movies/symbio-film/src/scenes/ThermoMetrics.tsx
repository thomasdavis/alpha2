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
import { metrics, sampleMetrics } from "../data";

// Animated area chart for sparse metrics
const SparseChart: React.FC<{
  data: { step: number; value: number }[];
  color: string;
  label: string;
  delay: number;
  width?: number;
  height?: number;
  decimals?: number;
  suffix?: string;
}> = ({
  data,
  color,
  label,
  delay,
  width = 780,
  height = 200,
  decimals = 4,
  suffix = "",
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const cardOpacity = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  const drawProgress = interpolate(frame - delay, [0, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  if (data.length < 2)
    return (
      <div
        style={{
          backgroundColor: THEME.bgCard,
          border: `1px solid ${THEME.border}`,
          borderRadius: 14,
          padding: 24,
          opacity: 0.4,
        }}
      >
        <div style={{ color: THEME.textDim, fontFamily, fontSize: 14 }}>
          {label}: No data
        </div>
      </div>
    );

  const yMin = Math.min(...data.map((d) => d.value));
  const yMax = Math.max(...data.map((d) => d.value));
  const range = yMax - yMin || 0.001;
  const padding = range * 0.1;

  const pts = data.map((d, i) => ({
    x: (i / (data.length - 1)) * width,
    y: height - ((d.value - yMin + padding) / (range + padding * 2)) * height,
  }));

  const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  const areaD = pathD + ` L ${width} ${height} L 0 ${height} Z`;

  const gradId = `sparseGrad-${label.replace(/\s/g, "")}`;

  const currentIdx = Math.min(
    data.length - 1,
    Math.floor(drawProgress * data.length)
  );
  const currentVal = data[currentIdx]?.value;

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
          marginBottom: 12,
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
            fontSize: 22,
            fontWeight: 700,
            color,
            fontFeatureSettings: '"tnum"',
            fontFamily,
          }}
        >
          {currentVal != null ? currentVal.toFixed(decimals) : "--"}
          {suffix}
        </span>
      </div>

      <svg width={width} height={height}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.25} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={t}
            x1={0}
            y1={t * height}
            x2={width}
            y2={t * height}
            stroke={THEME.gridLine}
            strokeWidth={0.5}
          />
        ))}
        <path d={areaD} fill={`url(#${gradId})`} opacity={drawProgress} />
        <path
          d={pathD}
          fill="none"
          stroke={color}
          strokeWidth={2.5}
          opacity={drawProgress}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 4px ${color}40)` }}
        />
        {drawProgress > 0.05 && (
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

// Population entropy chart
const PopulationEntropyChart: React.FC<{ delay: number }> = ({ delay }) => {
  const data = metrics
    .filter((m) => m.population_entropy != null)
    .map((m) => ({ step: m.step, value: m.population_entropy! }));

  return (
    <SparseChart
      data={data}
      color={THEME.pink}
      label="Population Entropy"
      delay={delay}
      decimals={3}
    />
  );
};

// Learning rate chart
const LRChart: React.FC<{ delay: number }> = ({ delay }) => {
  const sampled = sampleMetrics(200);
  const data = sampled.map((m) => ({ step: m.step, value: m.lr }));

  return (
    <SparseChart
      data={data}
      color={THEME.blue}
      label="Learning Rate"
      delay={delay}
      decimals={7}
    />
  );
};

export const ThermoMetricsScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  // Extract sparse metrics
  const weightEntropy = metrics
    .filter((m) => m.weight_entropy != null)
    .map((m) => ({ step: m.step, value: m.weight_entropy! }));

  const freeEnergy = metrics
    .filter((m) => m.free_energy != null)
    .map((m) => ({ step: m.step, value: m.free_energy! }));

  const fitnessScore = metrics
    .filter((m) => m.fitness_score != null)
    .map((m) => ({ step: m.step, value: m.fitness_score! }));

  const effectiveRank = metrics
    .filter((m) => m.effective_rank != null)
    .map((m) => ({ step: m.step, value: m.effective_rank! }));

  const fadeOut = interpolate(
    frame,
    [durationInFrames - 30, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SceneBackground variant="dark">
      <AbsoluteFill style={{ opacity: fadeOut, padding: "50px 80px" }}>
        <SceneTitle
          title="Thermodynamic Metrics"
          subtitle="Information-theoretic measures of training health and population dynamics"
        />

        {/* Stats row */}
        <div style={{ display: "flex", gap: 20, marginBottom: 24 }}>
          <StatCard
            label="Weight Entropy"
            value={weightEntropy.length > 0 ? weightEntropy[weightEntropy.length - 1].value.toFixed(4) : "N/A"}
            color={THEME.accent}
            delay={15}
          />
          <StatCard
            label="Effective Rank"
            value={effectiveRank.length > 0 ? effectiveRank[effectiveRank.length - 1].value.toFixed(1) : "N/A"}
            color={THEME.green}
            delay={20}
          />
          <StatCard
            label="Free Energy"
            value={freeEnergy.length > 0 ? freeEnergy[freeEnergy.length - 1].value.toFixed(3) : "N/A"}
            color={THEME.purple}
            delay={25}
          />
          <StatCard
            label="Fitness Score"
            value={fitnessScore.length > 0 ? fitnessScore[fitnessScore.length - 1].value.toFixed(4) : "N/A"}
            color={THEME.amber}
            delay={30}
          />
        </div>

        {/* Charts grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 20,
          }}
        >
          <SparseChart
            data={weightEntropy}
            color={THEME.accent}
            label="Weight Entropy"
            delay={50}
          />
          <SparseChart
            data={freeEnergy}
            color={THEME.purple}
            label="Free Energy"
            delay={70}
          />
          <Sequence from={100} layout="none">
            <SparseChart
              data={fitnessScore}
              color={THEME.amber}
              label="Fitness Score"
              delay={0}
            />
          </Sequence>
          <Sequence from={120} layout="none">
            <PopulationEntropyChart delay={0} />
          </Sequence>
          <Sequence from={160} layout="none">
            <SparseChart
              data={effectiveRank}
              color={THEME.green}
              label="Effective Rank"
              delay={0}
              decimals={1}
            />
          </Sequence>
          <Sequence from={180} layout="none">
            <LRChart delay={0} />
          </Sequence>
        </div>
      </AbsoluteFill>
    </SceneBackground>
  );
};
