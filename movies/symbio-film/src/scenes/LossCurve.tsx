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
  GlowDot,
} from "../components/shared";
import { metrics, getActivationColor, extractSwitchEvents, sampleMetrics } from "../data";

const CHART_LEFT = 160;
const CHART_TOP = 160;
const CHART_WIDTH = 1600;
const CHART_HEIGHT = 700;
const CHART_BOTTOM = CHART_TOP + CHART_HEIGHT;

function buildPath(
  data: { step: number; loss: number }[],
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number
): string {
  const pts = data.map((d) => {
    const x = CHART_LEFT + ((d.step - xMin) / (xMax - xMin)) * CHART_WIDTH;
    const y =
      CHART_BOTTOM - ((d.loss - yMin) / (yMax - yMin)) * CHART_HEIGHT;
    return { x, y };
  });
  return pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
}

const GridLines: React.FC<{
  yMin: number;
  yMax: number;
  xMin: number;
  xMax: number;
  opacity: number;
}> = ({ yMin, yMax, xMin, xMax, opacity }) => {
  const ySteps = 5;
  const xSteps = 6;
  const yLines = Array.from({ length: ySteps + 1 }, (_, i) => {
    const val = yMin + ((yMax - yMin) * i) / ySteps;
    const y = CHART_BOTTOM - ((val - yMin) / (yMax - yMin)) * CHART_HEIGHT;
    return { val, y };
  });
  const xLines = Array.from({ length: xSteps + 1 }, (_, i) => {
    const val = xMin + ((xMax - xMin) * i) / xSteps;
    const x = CHART_LEFT + ((val - xMin) / (xMax - xMin)) * CHART_WIDTH;
    return { val, x };
  });

  return (
    <g style={{ opacity }}>
      {yLines.map((l) => (
        <g key={`y-${l.val}`}>
          <line
            x1={CHART_LEFT}
            y1={l.y}
            x2={CHART_LEFT + CHART_WIDTH}
            y2={l.y}
            stroke={THEME.gridLine}
            strokeWidth={1}
          />
          <text
            x={CHART_LEFT - 16}
            y={l.y + 5}
            fill={THEME.textDim}
            fontSize={14}
            textAnchor="end"
            fontFamily={fontFamily}
          >
            {l.val.toFixed(1)}
          </text>
        </g>
      ))}
      {xLines.map((l) => (
        <g key={`x-${l.val}`}>
          <line
            x1={l.x}
            y1={CHART_TOP}
            x2={l.x}
            y2={CHART_BOTTOM}
            stroke={THEME.gridLine}
            strokeWidth={1}
          />
          <text
            x={l.x}
            y={CHART_BOTTOM + 28}
            fill={THEME.textDim}
            fontSize={14}
            textAnchor="middle"
            fontFamily={fontFamily}
          >
            {Math.round(l.val)}
          </text>
        </g>
      ))}
      {/* Axis labels */}
      <text
        x={CHART_LEFT + CHART_WIDTH / 2}
        y={CHART_BOTTOM + 55}
        fill={THEME.textMuted}
        fontSize={16}
        textAnchor="middle"
        fontFamily={fontFamily}
      >
        Training Step
      </text>
      <text
        x={CHART_LEFT - 60}
        y={CHART_TOP + CHART_HEIGHT / 2}
        fill={THEME.textMuted}
        fontSize={16}
        textAnchor="middle"
        fontFamily={fontFamily}
        transform={`rotate(-90, ${CHART_LEFT - 60}, ${CHART_TOP + CHART_HEIGHT / 2})`}
      >
        Loss
      </text>
    </g>
  );
};

export const LossCurveScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const sampled = sampleMetrics(400);
  const xMin = sampled[0].step;
  const xMax = sampled[sampled.length - 1].step;
  const yMin = 5.8;
  const yMax = 8.6;

  const switchEvents = extractSwitchEvents();

  // Build segmented paths by candidate
  const segments: {
    path: string;
    color: string;
    startIdx: number;
    endIdx: number;
  }[] = [];

  let segStart = 0;
  let prevAct = sampled[0]?.symbio_candidate_activation;
  for (let i = 1; i < sampled.length; i++) {
    const act = sampled[i].symbio_candidate_activation;
    if (act !== prevAct) {
      const segData = sampled.slice(segStart, i + 1);
      segments.push({
        path: buildPath(segData, xMin, xMax, yMin, yMax),
        color: getActivationColor(prevAct || null),
        startIdx: segStart,
        endIdx: i,
      });
      segStart = i;
      prevAct = act;
    }
  }
  // Last segment
  const segData = sampled.slice(segStart);
  segments.push({
    path: buildPath(segData, xMin, xMax, yMin, yMax),
    color: getActivationColor(prevAct || null),
    startIdx: segStart,
    endIdx: sampled.length - 1,
  });

  // Draw progress (0->1 over the main draw phase)
  const drawDuration = durationInFrames - 120;
  const drawProgress = interpolate(frame - 60, [0, drawDuration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  // Grid fade
  const gridOpacity = interpolate(frame, [0, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Fade out
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 30, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Current data point for tracking dot
  const currentIdx = Math.min(
    sampled.length - 1,
    Math.floor(drawProgress * sampled.length)
  );
  const currentPoint = sampled[currentIdx];
  const dotX =
    CHART_LEFT +
    ((currentPoint.step - xMin) / (xMax - xMin)) * CHART_WIDTH;
  const dotY =
    CHART_BOTTOM -
    ((currentPoint.loss - yMin) / (yMax - yMin)) * CHART_HEIGHT;

  return (
    <SceneBackground>
      <AbsoluteFill style={{ opacity: fadeOut, padding: 80 }}>
        <SceneTitle
          title="Training Loss"
          subtitle="Loss descent across 1,020 training steps with evolutionary activation search"
        />

        <svg
          width={1920}
          height={1080}
          style={{ position: "absolute", top: 0, left: 0 }}
        >
          <defs>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={THEME.accent} stopOpacity={0.2} />
              <stop offset="100%" stopColor={THEME.accent} stopOpacity={0} />
            </linearGradient>
          </defs>

          <GridLines
            yMin={yMin}
            yMax={yMax}
            xMin={xMin}
            xMax={xMax}
            opacity={gridOpacity}
          />

          {/* Activation switch lines */}
          {switchEvents.map((ev, i) => {
            const x =
              CHART_LEFT +
              ((ev.step - xMin) / (xMax - xMin)) * CHART_WIDTH;
            const evProgress = interpolate(
              drawProgress,
              [(ev.step - xMin) / (xMax - xMin) - 0.01, (ev.step - xMin) / (xMax - xMin) + 0.01],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );
            if (x < CHART_LEFT || x > CHART_LEFT + CHART_WIDTH) return null;
            return (
              <g key={i} style={{ opacity: evProgress * 0.3 }}>
                <line
                  x1={x}
                  y1={CHART_TOP}
                  x2={x}
                  y2={CHART_BOTTOM}
                  stroke={getActivationColor(ev.activation)}
                  strokeWidth={1}
                  strokeDasharray="4,4"
                />
              </g>
            );
          })}

          {/* Loss curve segments */}
          {segments.map((seg, i) => {
            const segStartRatio = seg.startIdx / sampled.length;
            const segEndRatio = seg.endIdx / sampled.length;
            const segProgress = interpolate(
              drawProgress,
              [segStartRatio, segEndRatio],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            if (segProgress <= 0) return null;

            const evolved = evolvePath(Math.min(segProgress, 1), seg.path);

            return (
              <path
                key={i}
                d={seg.path}
                fill="none"
                stroke={seg.color}
                strokeWidth={3}
                strokeDasharray={evolved.strokeDasharray}
                strokeDashoffset={evolved.strokeDashoffset}
                strokeLinecap="round"
                style={{
                  filter: `drop-shadow(0 0 6px ${seg.color}60)`,
                }}
              />
            );
          })}

          {/* Tracking dot */}
          {drawProgress > 0 && (
            <g>
              <circle
                cx={dotX}
                cy={dotY}
                r={8}
                fill={getActivationColor(currentPoint.symbio_candidate_activation)}
                style={{
                  filter: `drop-shadow(0 0 12px ${getActivationColor(currentPoint.symbio_candidate_activation)}80)`,
                }}
              />
              <circle
                cx={dotX}
                cy={dotY}
                r={4}
                fill="white"
              />
            </g>
          )}

          {/* Current value label */}
          {drawProgress > 0.05 && (
            <g>
              <rect
                x={dotX + 15}
                y={dotY - 30}
                width={140}
                height={40}
                rx={8}
                fill={`${THEME.bgCard}ee`}
                stroke={THEME.border}
              />
              <text
                x={dotX + 85}
                y={dotY - 5}
                fill={THEME.text}
                fontSize={18}
                fontWeight={600}
                textAnchor="middle"
                fontFamily={fontFamily}
              >
                {currentPoint.loss.toFixed(4)}
              </text>
            </g>
          )}
        </svg>

        {/* Legend */}
        <Sequence from={90} layout="none">
          <div
            style={{
              position: "absolute",
              top: 115,
              right: 100,
              display: "flex",
              gap: 16,
              flexWrap: "wrap",
              maxWidth: 500,
            }}
          >
            {["silu", "relu", "gelu", "composed"].map((act, i) => {
              const p = spring({
                frame: useCurrentFrame() - i * 5,
                fps,
                config: { damping: 200 },
              });
              return (
                <div
                  key={act}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    opacity: p,
                  }}
                >
                  <GlowDot
                    color={
                      act === "composed"
                        ? THEME.amber
                        : getActivationColor(act)
                    }
                    size={8}
                  />
                  <span
                    style={{
                      fontSize: 14,
                      color: THEME.textMuted,
                      fontFamily,
                    }}
                  >
                    {act}
                  </span>
                </div>
              );
            })}
          </div>
        </Sequence>

        {/* Step counter */}
        {drawProgress > 0 && (
          <div
            style={{
              position: "absolute",
              bottom: 50,
              right: 100,
              fontSize: 20,
              color: THEME.textDim,
              fontFeatureSettings: '"tnum"',
              fontFamily,
            }}
          >
            Step {currentPoint.step} / {metrics[metrics.length - 1].step}
            {"  "}
            <span style={{ color: getActivationColor(currentPoint.symbio_candidate_activation) }}>
              Gen {currentPoint.symbio_generation}
            </span>
            {"  "}
            <span style={{ color: THEME.textMuted, fontSize: 16 }}>
              {currentPoint.symbio_candidate_name}
            </span>
          </div>
        )}
      </AbsoluteFill>
    </SceneBackground>
  );
};
