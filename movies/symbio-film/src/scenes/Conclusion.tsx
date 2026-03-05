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
  StatCard,
} from "../components/shared";
import { metrics, run, fmtNum, getActivationColor, extractSwitchEvents } from "../data";

export const ConclusionScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const finalLoss = metrics[metrics.length - 1].loss;
  const startLoss = metrics[0].loss;
  const reduction = ((startLoss - finalLoss) / startLoss) * 100;

  const switchEvents = extractSwitchEvents();
  const finalActivation =
    metrics[metrics.length - 1].symbio_candidate_activation || "";
  const finalGeneration = metrics[metrics.length - 1].symbio_generation || 0;

  const uniqueActivations = new Set(
    metrics.map((m) => m.symbio_candidate_activation).filter(Boolean)
  );

  // Phase 1: "Training Complete" entrance
  const titleScale = spring({
    frame: frame - 20,
    fps,
    config: { damping: 15, stiffness: 80 },
    durationInFrames: 40,
  });

  const titleGlow = interpolate(frame, [40, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Final results grid
  const resultCards = [
    {
      label: "Final Loss",
      value: finalLoss.toFixed(4),
      color: THEME.accent,
      sub: `${reduction.toFixed(1)}% reduction`,
    },
    {
      label: "Start Loss",
      value: startLoss.toFixed(4),
      color: THEME.textMuted,
      sub: "initial",
    },
    {
      label: "Winner Activation",
      value: finalActivation.length > 22
        ? finalActivation.slice(0, 20) + "..."
        : finalActivation,
      color: getActivationColor(finalActivation),
      sub: `Generation ${finalGeneration}`,
    },
    {
      label: "Parameters",
      value: fmtNum(run.estimated_params),
      color: THEME.purple,
      sub: `${run.n_layer} layers, ${run.n_embd}d`,
    },
    {
      label: "Total Steps",
      value: String(metrics[metrics.length - 1].step),
      color: THEME.green,
      sub: `of ${run.total_iters} planned`,
    },
    {
      label: "Generations",
      value: String(finalGeneration + 1),
      color: THEME.amber,
      sub: `${switchEvents.length} candidate switches`,
    },
    {
      label: "Activations Explored",
      value: String(uniqueActivations.size),
      color: THEME.pink,
      sub: "unique compositions",
    },
    {
      label: "GPU",
      value: run.gpu_name || "GPU",
      color: THEME.blue,
      sub: `${(run.gpu_vram_mb / 1024).toFixed(0)}GB VRAM`,
    },
  ];

  // Animated loss bar
  const barProgress = interpolate(frame - 200, [0, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });

  // Fade out
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 45, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <SceneBackground variant="deep">
      <AbsoluteFill style={{ opacity: fadeOut }}>
        {/* Ambient particles */}
        {Array.from({ length: 20 }, (_, i) => {
          const angle = (i / 20) * Math.PI * 2 + frame * 0.005;
          const radius = 300 + Math.sin(frame * 0.02 + i) * 100;
          const x = 960 + Math.cos(angle) * radius;
          const y = 400 + Math.sin(angle) * radius;
          const opacity = interpolate(frame, [0, 60], [0, 0.3], {
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={i}
              style={{
                position: "absolute",
                left: x,
                top: y,
                width: 3,
                height: 3,
                borderRadius: "50%",
                backgroundColor: [THEME.accent, THEME.purple, THEME.green][
                  i % 3
                ],
                opacity,
                boxShadow: `0 0 8px ${[THEME.accent, THEME.purple, THEME.green][i % 3]}60`,
              }}
            />
          );
        })}

        {/* Title */}
        <div
          style={{
            position: "absolute",
            top: 120,
            left: 0,
            right: 0,
            textAlign: "center",
          }}
        >
          <div
            style={{
              fontSize: 80,
              fontWeight: 900,
              color: THEME.text,
              letterSpacing: "0.08em",
              transform: `scale(${titleScale})`,
              textShadow: `0 0 ${titleGlow * 40}px ${THEME.accent}40`,
              fontFamily,
            }}
          >
            TRAINING COMPLETE
          </div>
          <Sequence from={60} layout="none">
            <FadeIn delay={0}>
              <div
                style={{
                  fontSize: 24,
                  color: THEME.textMuted,
                  marginTop: 16,
                  fontFamily,
                }}
              >
                Symbiogenesis evolutionary activation search concluded
              </div>
            </FadeIn>
          </Sequence>
        </div>

        {/* Results grid */}
        <div
          style={{
            position: "absolute",
            top: 320,
            left: 100,
            right: 100,
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 20,
          }}
        >
          {resultCards.map((card, i) => (
            <Sequence key={card.label} from={100 + i * 8} layout="none">
              <StatCard
                label={card.label}
                value={card.value}
                color={card.color}
                delay={0}
                sub={card.sub}
              />
            </Sequence>
          ))}
        </div>

        {/* Loss reduction bar */}
        <Sequence from={200} layout="none">
          <div
            style={{
              position: "absolute",
              bottom: 200,
              left: 200,
              right: 200,
            }}
          >
            <FadeIn delay={0}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: 12,
                  fontFamily,
                }}
              >
                <span style={{ color: THEME.textMuted, fontSize: 14 }}>
                  Loss Reduction
                </span>
                <span
                  style={{
                    color: THEME.accent,
                    fontSize: 18,
                    fontWeight: 700,
                    fontFeatureSettings: '"tnum"',
                  }}
                >
                  {(reduction * barProgress).toFixed(1)}%
                </span>
              </div>
              <div
                style={{
                  height: 12,
                  backgroundColor: THEME.bgCard,
                  borderRadius: 6,
                  overflow: "hidden",
                  border: `1px solid ${THEME.border}`,
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${barProgress * reduction}%`,
                    background: `linear-gradient(90deg, ${THEME.accentDim}, ${THEME.accent})`,
                    borderRadius: 6,
                    boxShadow: `0 0 20px ${THEME.accent}40`,
                  }}
                />
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginTop: 8,
                  fontFamily,
                }}
              >
                <span style={{ color: THEME.textDim, fontSize: 12 }}>
                  {startLoss.toFixed(4)}
                </span>
                <span style={{ color: THEME.textDim, fontSize: 12 }}>
                  {finalLoss.toFixed(4)}
                </span>
              </div>
            </FadeIn>
          </div>
        </Sequence>

        {/* Footer */}
        <Sequence from={280} layout="none">
          <FadeIn delay={0}>
            <div
              style={{
                position: "absolute",
                bottom: 60,
                left: 0,
                right: 0,
                textAlign: "center",
                fontSize: 16,
                color: THEME.textDim,
                fontFamily,
              }}
            >
              {run.hostname} | {run.gpu_name} | {run.os_platform} | Run {run.run_id}
            </div>
          </FadeIn>
        </Sequence>
      </AbsoluteFill>
    </SceneBackground>
  );
};
