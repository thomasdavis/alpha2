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
import { THEME } from "../theme";
import {
  fontFamily,
  SceneBackground,
  SceneTitle,
  FadeIn,
  GlowDot,
  Badge,
} from "../components/shared";
import { metrics, extractSwitchEvents, getActivationColor } from "../data";

type CandidateInfo = {
  name: string;
  activation: string;
  generation: number;
  parent: string | null;
  steps: number;
  bestLoss: number;
  avgLoss: number;
  color: string;
};

function extractCandidates(): CandidateInfo[] {
  const map = new Map<string, { losses: number[]; m: typeof metrics[0] }>();
  for (const m of metrics) {
    const name = m.symbio_candidate_name;
    if (!name) continue;
    const entry = map.get(name) || { losses: [], m };
    entry.losses.push(m.loss);
    entry.m = m;
    map.set(name, entry);
  }
  return Array.from(map.entries()).map(([name, { losses, m }]) => ({
    name,
    activation: m.symbio_candidate_activation || "",
    generation: m.symbio_generation || 0,
    parent: m.symbio_candidate_parent_name || null,
    steps: losses.length,
    bestLoss: Math.min(...losses),
    avgLoss: losses.reduce((a, b) => a + b, 0) / losses.length,
    color: getActivationColor(m.symbio_candidate_activation),
  }));
}

// Tree node in the lineage
const TreeNode: React.FC<{
  candidate: CandidateInfo;
  x: number;
  y: number;
  progress: number;
  isWinner?: boolean;
}> = ({ candidate, x, y, progress, isWinner }) => {
  const radius = Math.max(12, Math.min(30, 40 - candidate.bestLoss * 3));

  return (
    <g style={{ opacity: progress }}>
      {/* Glow for winner */}
      {isWinner && (
        <circle
          cx={x}
          cy={y}
          r={radius + 12}
          fill="none"
          stroke={THEME.amber}
          strokeWidth={2}
          opacity={0.5}
          strokeDasharray="4,4"
        />
      )}
      {/* Node circle */}
      <circle
        cx={x}
        cy={y}
        r={radius * progress}
        fill={candidate.color}
        opacity={0.8}
        style={{ filter: `drop-shadow(0 0 8px ${candidate.color}60)` }}
      />
      <circle
        cx={x}
        cy={y}
        r={radius * 0.4 * progress}
        fill="white"
        opacity={0.3}
      />
      {/* Label */}
      <text
        x={x}
        y={y + radius + 18}
        fill={THEME.textMuted}
        fontSize={10}
        textAnchor="middle"
        fontFamily={fontFamily}
        opacity={progress}
      >
        {candidate.activation.length > 15
          ? candidate.activation.slice(0, 12) + "..."
          : candidate.activation}
      </text>
    </g>
  );
};

// Generation timeline strip
const GenerationStrip: React.FC<{
  generation: number;
  candidates: CandidateInfo[];
  delay: number;
  y: number;
}> = ({ generation, candidates, delay, y }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const stripProgress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
    durationInFrames: 30,
  });

  const sorted = [...candidates].sort((a, b) => a.bestLoss - b.bestLoss);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 16,
        opacity: stripProgress,
        transform: `translateX(${(1 - stripProgress) * 60}px)`,
      }}
    >
      {/* Gen label */}
      <div
        style={{
          width: 80,
          textAlign: "right",
          fontSize: 15,
          fontWeight: 600,
          color: THEME.accent,
          fontFamily,
          flexShrink: 0,
        }}
      >
        Gen {generation}
      </div>

      {/* Candidates */}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {sorted.map((c, i) => {
          const cardProgress = spring({
            frame: frame - delay - i * 4,
            fps,
            config: { damping: 18, stiffness: 100 },
          });

          const isLeader = i === 0;

          return (
            <div
              key={c.name}
              style={{
                backgroundColor: isLeader ? `${c.color}15` : THEME.bgCard,
                border: `1px solid ${isLeader ? c.color + "50" : THEME.border}`,
                borderRadius: 10,
                padding: "10px 16px",
                opacity: cardProgress,
                transform: `scale(${0.85 + cardProgress * 0.15})`,
                minWidth: 140,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <GlowDot color={c.color} size={8} />
                <span
                  style={{
                    fontSize: 12,
                    color: THEME.text,
                    fontWeight: 600,
                    fontFamily,
                    maxWidth: 120,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {c.activation.length > 18
                    ? c.activation.slice(0, 15) + "..."
                    : c.activation}
                </span>
              </div>
              <div
                style={{
                  fontSize: 20,
                  fontWeight: 700,
                  color: c.color,
                  fontFeatureSettings: '"tnum"',
                  fontFamily,
                }}
              >
                {c.bestLoss.toFixed(3)}
              </div>
              <div
                style={{
                  fontSize: 10,
                  color: THEME.textDim,
                  marginTop: 2,
                  fontFamily,
                }}
              >
                {c.steps} steps
                {c.parent ? ` | from ${c.parent.split("-")[0]}` : ""}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Diversity area chart
const DiversityChart: React.FC<{ delay: number }> = ({ delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const divData = metrics
    .filter((m) => m.architecture_diversity != null)
    .filter((_, i) => i % 5 === 0);

  const width = 800;
  const height = 120;

  const progress = interpolate(frame - delay, [0, 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const cardOpacity = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  const pts = divData.map((m, i) => ({
    x: (i / (divData.length - 1)) * width,
    y: height - (m.architecture_diversity || 0) * height,
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
        Architecture Diversity
      </div>
      <svg width={width} height={height}>
        <defs>
          <linearGradient id="divGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={THEME.purple} stopOpacity={0.3} />
            <stop offset="100%" stopColor={THEME.purple} stopOpacity={0} />
          </linearGradient>
        </defs>
        <path d={areaD} fill="url(#divGrad)" opacity={progress} />
        <path
          d={pathD}
          fill="none"
          stroke={THEME.purple}
          strokeWidth={2}
          opacity={progress}
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
};

export const EvolutionScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const candidates = extractCandidates();
  const byGen = new Map<number, CandidateInfo[]>();
  for (const c of candidates) {
    const list = byGen.get(c.generation) || [];
    list.push(c);
    byGen.set(c.generation, list);
  }

  const generations = Array.from(byGen.keys()).sort((a, b) => a - b);

  const fadeOut = interpolate(
    frame,
    [durationInFrames - 30, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const totalCandidates = candidates.length;
  const bestCandidate = candidates.reduce((best, c) =>
    c.bestLoss < best.bestLoss ? c : best
  );

  return (
    <SceneBackground>
      <AbsoluteFill style={{ opacity: fadeOut, padding: "50px 80px" }}>
        <SceneTitle
          title="Evolutionary Search"
          subtitle={`${totalCandidates} candidates across ${generations.length} generations`}
        />

        {/* Stats */}
        <Sequence from={30} layout="none">
          <div style={{ display: "flex", gap: 16, marginBottom: 24 }}>
            <Badge text={`${generations.length} Generations`} color={THEME.accent} />
            <Badge text={`${totalCandidates} Candidates`} color={THEME.purple} />
            <Badge
              text={`Winner: ${bestCandidate.activation}`}
              color={THEME.amber}
            />
            <Badge text={`Best: ${bestCandidate.bestLoss.toFixed(3)}`} color={THEME.green} />
          </div>
        </Sequence>

        {/* Generation timeline */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 14,
            flex: 1,
            overflowY: "hidden",
          }}
        >
          {generations.map((gen, i) => (
            <GenerationStrip
              key={gen}
              generation={gen}
              candidates={byGen.get(gen) || []}
              delay={60 + i * 20}
              y={0}
            />
          ))}
        </div>

        {/* Diversity chart at bottom */}
        <Sequence from={durationInFrames - 300} layout="none">
          <div style={{ marginTop: 16 }}>
            <DiversityChart delay={0} />
          </div>
        </Sequence>
      </AbsoluteFill>
    </SceneBackground>
  );
};
