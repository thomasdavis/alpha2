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
  FadeIn,
  GlowDot,
} from "../components/shared";
import { metrics, sampleMetrics, getActivationColor, extractSwitchEvents } from "../data";

const CX = 960;
const CY = 500;
const OUTER_R = 380;
const LOSS_R = 300;
const FITNESS_R = 240;
const DIVERSITY_R = 190;
const INNER_R = 120;

export const RadialVizScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const sampled = sampleMetrics(360);
  const totalPoints = sampled.length;
  const switchEvents = extractSwitchEvents();

  // Build phase
  const buildProgress = interpolate(frame, [30, durationInFrames - 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  const ringFade = spring({
    frame: frame - 10,
    fps,
    config: { damping: 200 },
    durationInFrames: 30,
  });

  const fadeOut = interpolate(
    frame,
    [durationInFrames - 30, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Rotation
  const rotation = frame * 0.05;

  // Loss range for normalization
  const losses = sampled.map((m) => m.loss);
  const lossMin = Math.min(...losses);
  const lossMax = Math.max(...losses);
  const lossRange = lossMax - lossMin || 1;

  // Fitness range
  const fitnessPts = sampled.filter((m) => m.fitness_score != null);
  const fitnessMin = fitnessPts.length > 0 ? Math.min(...fitnessPts.map((m) => m.fitness_score!)) : 0;
  const fitnessMax = fitnessPts.length > 0 ? Math.max(...fitnessPts.map((m) => m.fitness_score!)) : 1;
  const fitnessRange = fitnessMax - fitnessMin || 1;

  // Draw the visible portion
  const visibleCount = Math.floor(buildProgress * totalPoints);

  // Current point
  const currentIdx = Math.max(0, visibleCount - 1);
  const current = sampled[currentIdx];

  return (
    <SceneBackground variant="deep">
      <AbsoluteFill style={{ opacity: fadeOut }}>
        {/* Title */}
        <Sequence from={0} layout="none">
          <FadeIn delay={0}>
            <div
              style={{
                position: "absolute",
                top: 40,
                left: 80,
                fontSize: 48,
                fontWeight: 700,
                color: THEME.text,
                fontFamily,
                letterSpacing: "-0.02em",
              }}
            >
              Radial Training Overview
            </div>
            <div
              style={{
                position: "absolute",
                top: 100,
                left: 80,
                fontSize: 18,
                color: THEME.textMuted,
                fontFamily,
              }}
            >
              Loss, fitness, diversity, and candidate evolution mapped to polar coordinates
            </div>
          </FadeIn>
        </Sequence>

        <svg
          width={1920}
          height={1080}
          style={{ position: "absolute", top: 0, left: 0 }}
        >
          <g transform={`rotate(${rotation}, ${CX}, ${CY})`}>
            {/* Guide rings */}
            {[OUTER_R, LOSS_R, FITNESS_R, DIVERSITY_R, INNER_R].map((r) => (
              <circle
                key={r}
                cx={CX}
                cy={CY}
                r={r}
                fill="none"
                stroke={THEME.gridLine}
                strokeWidth={0.5}
                opacity={ringFade * 0.4}
              />
            ))}

            {/* Outer band: activation segments */}
            {sampled.slice(0, visibleCount).map((m, i) => {
              const angle = (i / totalPoints) * Math.PI * 2 - Math.PI / 2;
              const nextAngle =
                ((i + 1) / totalPoints) * Math.PI * 2 - Math.PI / 2;
              const color = getActivationColor(m.symbio_candidate_activation);

              const x1 = CX + Math.cos(angle) * (OUTER_R - 15);
              const y1 = CY + Math.sin(angle) * (OUTER_R - 15);
              const x2 = CX + Math.cos(angle) * OUTER_R;
              const y2 = CY + Math.sin(angle) * OUTER_R;

              return (
                <line
                  key={`act-${i}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={color}
                  strokeWidth={3}
                  opacity={0.7}
                />
              );
            })}

            {/* Loss ring */}
            {sampled.slice(0, visibleCount).map((m, i) => {
              if (i === 0) return null;
              const angle = (i / totalPoints) * Math.PI * 2 - Math.PI / 2;
              const normLoss = 1 - (m.loss - lossMin) / lossRange;
              const r = INNER_R + normLoss * (LOSS_R - INNER_R);
              const color = getActivationColor(m.symbio_candidate_activation);

              const prevAngle =
                ((i - 1) / totalPoints) * Math.PI * 2 - Math.PI / 2;
              const prevM = sampled[i - 1];
              const prevNorm = 1 - (prevM.loss - lossMin) / lossRange;
              const prevR = INNER_R + prevNorm * (LOSS_R - INNER_R);

              return (
                <line
                  key={`loss-${i}`}
                  x1={CX + Math.cos(prevAngle) * prevR}
                  y1={CY + Math.sin(prevAngle) * prevR}
                  x2={CX + Math.cos(angle) * r}
                  y2={CY + Math.sin(angle) * r}
                  stroke={color}
                  strokeWidth={2}
                  opacity={0.8}
                  style={{ filter: `drop-shadow(0 0 3px ${color}60)` }}
                />
              );
            })}

            {/* Fitness dots (sparse) */}
            {sampled.slice(0, visibleCount).map((m, i) => {
              if (m.fitness_score == null) return null;
              const angle = (i / totalPoints) * Math.PI * 2 - Math.PI / 2;
              const normFit = (m.fitness_score - fitnessMin) / fitnessRange;
              const r = FITNESS_R - 5 + normFit * 30;
              const color = getActivationColor(m.symbio_candidate_activation);

              return (
                <circle
                  key={`fit-${i}`}
                  cx={CX + Math.cos(angle) * r}
                  cy={CY + Math.sin(angle) * r}
                  r={3}
                  fill={color}
                  opacity={0.6}
                />
              );
            })}

            {/* Diversity ring */}
            {sampled.slice(0, visibleCount).map((m, i) => {
              if (i === 0 || m.architecture_diversity == null) return null;
              const angle = (i / totalPoints) * Math.PI * 2 - Math.PI / 2;
              const r = DIVERSITY_R - 10 + (m.architecture_diversity || 0) * 20;

              const prevAngle =
                ((i - 1) / totalPoints) * Math.PI * 2 - Math.PI / 2;
              const prevM = sampled[i - 1];
              const prevR =
                DIVERSITY_R - 10 + (prevM.architecture_diversity || 0) * 20;

              return (
                <line
                  key={`div-${i}`}
                  x1={CX + Math.cos(prevAngle) * prevR}
                  y1={CY + Math.sin(prevAngle) * prevR}
                  x2={CX + Math.cos(angle) * r}
                  y2={CY + Math.sin(angle) * r}
                  stroke={THEME.purple}
                  strokeWidth={1.5}
                  opacity={0.4}
                />
              );
            })}

            {/* Switch event markers */}
            {switchEvents.map((ev, i) => {
              const idx = sampled.findIndex((m) => m.step >= ev.step);
              if (idx < 0 || idx >= visibleCount) return null;
              const angle = (idx / totalPoints) * Math.PI * 2 - Math.PI / 2;
              const x = CX + Math.cos(angle) * (OUTER_R + 10);
              const y = CY + Math.sin(angle) * (OUTER_R + 10);
              const color = getActivationColor(ev.activation);

              return (
                <g key={`sw-${i}`}>
                  <line
                    x1={CX + Math.cos(angle) * INNER_R}
                    y1={CY + Math.sin(angle) * INNER_R}
                    x2={CX + Math.cos(angle) * OUTER_R}
                    y2={CY + Math.sin(angle) * OUTER_R}
                    stroke={color}
                    strokeWidth={0.5}
                    opacity={0.3}
                    strokeDasharray="2,4"
                  />
                </g>
              );
            })}

            {/* Current position dot */}
            {visibleCount > 0 && (
              <>
                {(() => {
                  const angle =
                    (currentIdx / totalPoints) * Math.PI * 2 - Math.PI / 2;
                  const normLoss = 1 - (current.loss - lossMin) / lossRange;
                  const r = INNER_R + normLoss * (LOSS_R - INNER_R);
                  const color = getActivationColor(
                    current.symbio_candidate_activation
                  );
                  const x = CX + Math.cos(angle) * r;
                  const y = CY + Math.sin(angle) * r;

                  return (
                    <g>
                      <circle
                        cx={x}
                        cy={y}
                        r={10}
                        fill={color}
                        style={{
                          filter: `drop-shadow(0 0 15px ${color}aa)`,
                        }}
                      />
                      <circle cx={x} cy={y} r={4} fill="white" opacity={0.8} />
                    </g>
                  );
                })()}
              </>
            )}
          </g>

          {/* Center text (not rotated) */}
          <text
            x={CX}
            y={CY - 20}
            fill={THEME.text}
            fontSize={28}
            fontWeight={700}
            textAnchor="middle"
            fontFamily={fontFamily}
          >
            {current?.symbio_candidate_activation || ""}
          </text>
          <text
            x={CX}
            y={CY + 15}
            fill={THEME.accent}
            fontSize={36}
            fontWeight={800}
            textAnchor="middle"
            fontFamily={fontFamily}
          >
            {current?.loss.toFixed(4) || ""}
          </text>
          <text
            x={CX}
            y={CY + 45}
            fill={THEME.textDim}
            fontSize={14}
            textAnchor="middle"
            fontFamily={fontFamily}
          >
            Step {current?.step || 0} | Gen {current?.symbio_generation || 0}
          </text>
        </svg>

        {/* Legend */}
        <Sequence from={60} layout="none">
          <FadeIn delay={0}>
            <div
              style={{
                position: "absolute",
                bottom: 60,
                right: 80,
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              {[
                { label: "Outer Band", desc: "Active Activation", color: THEME.accent },
                { label: "Loss Ring", desc: "Normalized Loss", color: THEME.green },
                { label: "Fitness Ring", desc: "Candidate Fitness", color: THEME.amber },
                { label: "Diversity Ring", desc: "Architecture Diversity", color: THEME.purple },
              ].map((item) => (
                <div
                  key={item.label}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    fontSize: 13,
                    fontFamily,
                  }}
                >
                  <div
                    style={{
                      width: 30,
                      height: 3,
                      backgroundColor: item.color,
                      borderRadius: 2,
                    }}
                  />
                  <span style={{ color: THEME.text, fontWeight: 500 }}>
                    {item.label}
                  </span>
                  <span style={{ color: THEME.textDim }}>
                    {item.desc}
                  </span>
                </div>
              ))}
            </div>
          </FadeIn>
        </Sequence>

        {/* Activation color key */}
        <Sequence from={90} layout="none">
          <FadeIn delay={0}>
            <div
              style={{
                position: "absolute",
                bottom: 60,
                left: 80,
                display: "flex",
                gap: 16,
                flexWrap: "wrap",
                maxWidth: 600,
              }}
            >
              {["silu", "relu", "gelu", "id", "sq"].map((act) => (
                <div
                  key={act}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: 13,
                    fontFamily,
                  }}
                >
                  <GlowDot color={getActivationColor(act)} size={8} />
                  <span style={{ color: THEME.textMuted }}>{act}</span>
                </div>
              ))}
            </div>
          </FadeIn>
        </Sequence>
      </AbsoluteFill>
    </SceneBackground>
  );
};
