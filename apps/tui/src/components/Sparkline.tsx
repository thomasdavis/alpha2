import React from "react";
import { Box, Text } from "ink";
import { sparkline } from "../lib/chart.js";

// Gradient colors from high (bad) to low (good) for loss sparklines
const GRADIENT_COLORS = ["greenBright", "green", "yellow", "red", "redBright"] as const;

function intensityColor(intensity: number): string {
  const idx = Math.round(intensity * (GRADIENT_COLORS.length - 1));
  return GRADIENT_COLORS[Math.max(0, Math.min(GRADIENT_COLORS.length - 1, idx))];
}

export function Sparkline({
  values,
  width = 20,
  gradient = false,
  color = "yellow",
}: {
  values: number[];
  width?: number;
  gradient?: boolean;
  color?: string;
}) {
  if (values.length === 0) {
    return <Text color="gray" dimColor>{"·".repeat(Math.min(width, 8))}</Text>;
  }

  const chars = sparkline(values, width);

  if (!gradient) {
    return (
      <Text>
        {chars.map((c, i) => (
          <Text key={i} color={color}>{c.char}</Text>
        ))}
      </Text>
    );
  }

  // Gradient mode: color each character by its relative value
  return (
    <Text>
      {chars.map((c, i) => (
        <Text key={i} color={intensityColor(c.intensity)}>{c.char}</Text>
      ))}
    </Text>
  );
}
