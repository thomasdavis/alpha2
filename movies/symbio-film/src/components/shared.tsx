import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";
import { loadFont } from "@remotion/google-fonts/Inter";
import { THEME } from "../theme";

const { fontFamily } = loadFont("normal", {
  weights: ["300", "400", "500", "600", "700", "800", "900"],
  subsets: ["latin"],
});

export { fontFamily };

// Animated background with subtle grid
export const SceneBackground: React.FC<{
  children: React.ReactNode;
  variant?: "default" | "dark" | "deep";
}> = ({ children, variant = "default" }) => {
  const frame = useCurrentFrame();
  const bgColor =
    variant === "deep"
      ? "#050508"
      : variant === "dark"
        ? "#070710"
        : THEME.bg;

  const gridOpacity = interpolate(frame, [0, 30], [0, 0.15], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: bgColor,
        fontFamily,
        overflow: "hidden",
      }}
    >
      {/* Subtle grid */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          opacity: gridOpacity,
          backgroundImage: `
            linear-gradient(${THEME.gridLine} 1px, transparent 1px),
            linear-gradient(90deg, ${THEME.gridLine} 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
        }}
      />
      {/* Ambient glow */}
      <div
        style={{
          position: "absolute",
          width: 800,
          height: 800,
          borderRadius: "50%",
          background: `radial-gradient(circle, rgba(34,211,238,0.06) 0%, transparent 70%)`,
          top: "50%",
          left: "50%",
          transform: `translate(-50%, -50%) scale(${1 + Math.sin(frame * 0.01) * 0.1})`,
        }}
      />
      {children}
    </AbsoluteFill>
  );
};

// Fade-in wrapper
export const FadeIn: React.FC<{
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  style?: React.CSSProperties;
}> = ({ children, delay = 0, duration = 20, style }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame - delay, [0, duration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const y = interpolate(frame - delay, [0, duration], [30, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });
  return (
    <div style={{ opacity, transform: `translateY(${y}px)`, ...style }}>
      {children}
    </div>
  );
};

// Scene title with animated underline
export const SceneTitle: React.FC<{
  title: string;
  subtitle?: string;
  delay?: number;
}> = ({ title, subtitle, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleProgress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  const lineWidth = spring({
    frame: frame - delay - 10,
    fps,
    config: { damping: 200 },
    durationInFrames: 30,
  });

  const subtitleOpacity = interpolate(frame - delay - 20, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ marginBottom: 40 }}>
      <div
        style={{
          fontSize: 56,
          fontWeight: 700,
          color: THEME.text,
          letterSpacing: "-0.02em",
          opacity: titleProgress,
          transform: `translateY(${(1 - titleProgress) * 20}px)`,
        }}
      >
        {title}
      </div>
      <div
        style={{
          height: 3,
          backgroundColor: THEME.accent,
          marginTop: 12,
          width: `${lineWidth * 120}px`,
          borderRadius: 2,
          boxShadow: `0 0 20px ${THEME.accent}40`,
        }}
      />
      {subtitle && (
        <div
          style={{
            fontSize: 22,
            color: THEME.textMuted,
            marginTop: 16,
            opacity: subtitleOpacity,
            fontWeight: 400,
          }}
        >
          {subtitle}
        </div>
      )}
    </div>
  );
};

// Stat card that animates in
export const StatCard: React.FC<{
  label: string;
  value: string;
  color?: string;
  delay?: number;
  sub?: string;
}> = ({ label, value, color = THEME.accent, delay = 0, sub }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
  });

  return (
    <div
      style={{
        backgroundColor: THEME.bgCard,
        border: `1px solid ${THEME.border}`,
        borderRadius: 12,
        padding: "20px 28px",
        opacity: progress,
        transform: `scale(${0.9 + progress * 0.1})`,
      }}
    >
      <div
        style={{
          fontSize: 14,
          color: THEME.textMuted,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          marginBottom: 8,
          fontWeight: 500,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 36,
          fontWeight: 700,
          color,
          fontFeatureSettings: '"tnum"',
        }}
      >
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 13, color: THEME.textDim, marginTop: 4 }}>
          {sub}
        </div>
      )}
    </div>
  );
};

// Animated counter
export const AnimatedNumber: React.FC<{
  value: number;
  delay?: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
}> = ({ value, delay = 0, duration = 60, decimals = 0, prefix = "", suffix = "" }) => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame - delay, [0, duration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.quad),
  });
  const current = value * progress;
  return (
    <span style={{ fontFeatureSettings: '"tnum"' }}>
      {prefix}
      {current.toFixed(decimals)}
      {suffix}
    </span>
  );
};

// Glowing dot
export const GlowDot: React.FC<{
  color: string;
  size?: number;
}> = ({ color, size = 10 }) => (
  <div
    style={{
      width: size,
      height: size,
      borderRadius: "50%",
      backgroundColor: color,
      boxShadow: `0 0 ${size}px ${color}80, 0 0 ${size * 2}px ${color}40`,
      flexShrink: 0,
    }}
  />
);

// Section label badge
export const Badge: React.FC<{
  text: string;
  color?: string;
}> = ({ text, color = THEME.accent }) => (
  <div
    style={{
      display: "inline-flex",
      padding: "4px 14px",
      borderRadius: 20,
      backgroundColor: `${color}15`,
      border: `1px solid ${color}40`,
      color,
      fontSize: 13,
      fontWeight: 600,
      letterSpacing: "0.05em",
      textTransform: "uppercase",
    }}
  >
    {text}
  </div>
);
