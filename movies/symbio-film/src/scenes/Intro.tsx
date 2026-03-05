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
import { fontFamily, SceneBackground, FadeIn, Badge } from "../components/shared";
import { run, fmtNum } from "../data";

// Floating particles
const Particle: React.FC<{
  x: number;
  y: number;
  size: number;
  speed: number;
  color: string;
  delay: number;
}> = ({ x, y, size, speed, color, delay }) => {
  const frame = useCurrentFrame();
  const t = (frame - delay) * speed * 0.01;
  const px = x + Math.sin(t * 1.3) * 40;
  const py = y + Math.cos(t) * 30 - frame * 0.3;
  const opacity = interpolate(frame - delay, [0, 30, 300], [0, 0.6, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <div
      style={{
        position: "absolute",
        left: px,
        top: py,
        width: size,
        height: size,
        borderRadius: "50%",
        backgroundColor: color,
        opacity,
        boxShadow: `0 0 ${size * 2}px ${color}60`,
      }}
    />
  );
};

const TypewriterText: React.FC<{
  text: string;
  delay: number;
  charFrames?: number;
}> = ({ text, delay, charFrames = 2 }) => {
  const frame = useCurrentFrame();
  const elapsed = Math.max(0, frame - delay);
  const chars = Math.min(text.length, Math.floor(elapsed / charFrames));
  const cursorOpacity = Math.sin(frame * 0.3) > 0 ? 1 : 0;

  return (
    <span>
      {text.slice(0, chars)}
      {chars < text.length && (
        <span style={{ opacity: cursorOpacity, color: THEME.accent }}>|</span>
      )}
    </span>
  );
};

export const IntroScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Phase 1: Logo/title entrance (frames 0-90)
  const titleScale = spring({
    frame,
    fps,
    config: { damping: 15, stiffness: 80 },
    durationInFrames: 40,
  });

  const titleGlow = interpolate(frame, [30, 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Rotating ring
  const ringRotation = frame * 0.5;
  const ringOpacity = interpolate(frame, [0, 40], [0, 0.4], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Phase 2: Subtitle (frames 60+)
  const subtitleProgress = spring({
    frame: frame - 60,
    fps,
    config: { damping: 200 },
  });

  // Phase 3: Model config cards (frames 120+)
  const configDelay = 120;

  // Phase 4: Description text (frames 200+)

  // Fade out
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 30, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const particles = Array.from({ length: 30 }, (_, i) => ({
    x: 200 + (i * 57) % 1500,
    y: 100 + (i * 83) % 900,
    size: 2 + (i % 4),
    speed: 0.5 + (i % 3) * 0.3,
    color: [THEME.accent, THEME.purple, THEME.green, THEME.pink][i % 4],
    delay: i * 5,
  }));

  const modelConfig = JSON.parse(run.model_config);

  return (
    <SceneBackground variant="deep">
      <AbsoluteFill style={{ opacity: fadeOut }}>
        {/* Particles */}
        {particles.map((p, i) => (
          <Particle key={i} {...p} />
        ))}

        {/* Central ring */}
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: `translate(-50%, -50%) rotate(${ringRotation}deg)`,
            width: 500,
            height: 500,
            borderRadius: "50%",
            border: `2px solid ${THEME.accent}30`,
            opacity: ringOpacity,
          }}
        />
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: `translate(-50%, -50%) rotate(${-ringRotation * 0.7}deg)`,
            width: 600,
            height: 600,
            borderRadius: "50%",
            border: `1px dashed ${THEME.purple}20`,
            opacity: ringOpacity * 0.6,
          }}
        />

        {/* Title: SYMBIO */}
        <div
          style={{
            position: "absolute",
            top: 180,
            left: 0,
            right: 0,
            textAlign: "center",
          }}
        >
          <div
            style={{
              fontSize: 140,
              fontWeight: 900,
              color: THEME.text,
              letterSpacing: "0.15em",
              transform: `scale(${titleScale})`,
              textShadow: `0 0 ${titleGlow * 60}px ${THEME.accent}60, 0 0 ${titleGlow * 120}px ${THEME.accent}20`,
              fontFamily,
            }}
          >
            SYMBIO
          </div>

          {/* Subtitle */}
          <div
            style={{
              fontSize: 28,
              color: THEME.textMuted,
              letterSpacing: "0.3em",
              textTransform: "uppercase",
              marginTop: 20,
              opacity: subtitleProgress,
              transform: `translateY(${(1 - subtitleProgress) * 15}px)`,
              fontWeight: 300,
            }}
          >
            Symbiogenesis Neural Architecture Search
          </div>
        </div>

        {/* Model config grid */}
        <Sequence from={configDelay} layout="none">
          <div
            style={{
              position: "absolute",
              bottom: 240,
              left: 140,
              right: 140,
              display: "flex",
              gap: 20,
              justifyContent: "center",
            }}
          >
            {[
              { label: "Parameters", value: fmtNum(run.estimated_params), color: THEME.accent },
              { label: "Layers", value: String(modelConfig.nLayer), color: THEME.purple },
              { label: "Embedding", value: String(modelConfig.nEmbd), color: THEME.green },
              { label: "Heads", value: String(modelConfig.nHead), color: THEME.amber },
              { label: "Vocab", value: fmtNum(run.vocab_size), color: THEME.pink },
              { label: "Context", value: String(run.block_size), color: THEME.orange },
              { label: "Activation", value: modelConfig.ffnActivation.toUpperCase(), color: THEME.accent },
              { label: "Backend", value: run.backend.toUpperCase(), color: THEME.blue },
            ].map((item, i) => {
              const p = spring({
                frame: useCurrentFrame() - i * 5,
                fps,
                config: { damping: 200 },
              });
              return (
                <div
                  key={item.label}
                  style={{
                    backgroundColor: `${THEME.bgCard}ee`,
                    border: `1px solid ${THEME.border}`,
                    borderRadius: 10,
                    padding: "16px 22px",
                    textAlign: "center",
                    opacity: p,
                    transform: `translateY(${(1 - p) * 30}px)`,
                    minWidth: 130,
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      color: THEME.textMuted,
                      textTransform: "uppercase",
                      letterSpacing: "0.1em",
                      marginBottom: 6,
                      fontWeight: 500,
                    }}
                  >
                    {item.label}
                  </div>
                  <div
                    style={{
                      fontSize: 28,
                      fontWeight: 700,
                      color: item.color,
                      fontFeatureSettings: '"tnum"',
                    }}
                  >
                    {item.value}
                  </div>
                </div>
              );
            })}
          </div>
        </Sequence>

        {/* Description */}
        <Sequence from={200} layout="none">
          <div
            style={{
              position: "absolute",
              bottom: 90,
              left: 0,
              right: 0,
              textAlign: "center",
            }}
          >
            <FadeIn delay={0}>
              <div
                style={{
                  fontSize: 18,
                  color: THEME.textMuted,
                  maxWidth: 900,
                  margin: "0 auto",
                  lineHeight: 1.6,
                }}
              >
                <TypewriterText
                  text="Evolving activation functions through competitive selection across 9 generations — searching for the optimal neural architecture."
                  delay={0}
                  charFrames={1}
                />
              </div>
            </FadeIn>
          </div>
        </Sequence>

        {/* GPU badge */}
        <Sequence from={160} layout="none">
          <FadeIn delay={0}>
            <div
              style={{
                position: "absolute",
                top: 60,
                right: 80,
                display: "flex",
                gap: 12,
                alignItems: "center",
              }}
            >
              <Badge text={run.gpu_name || "GPU"} color={THEME.green} />
              <Badge text={`${(run.gpu_vram_mb / 1024).toFixed(0)}GB VRAM`} color={THEME.amber} />
            </div>
          </FadeIn>
        </Sequence>
      </AbsoluteFill>
    </SceneBackground>
  );
};
