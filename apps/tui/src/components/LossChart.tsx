import React from "react";
import { Box, Text } from "ink";
import type { MetricPoint } from "../types.js";
import { renderChart } from "../lib/chart.js";

export function LossChart({
  metrics,
  width = 50,
  height = 8,
}: {
  metrics: MetricPoint[];
  width?: number;
  height?: number;
}) {
  if (metrics.length === 0) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text color="gray" dimColor>
          {"  ╭" + "─".repeat(width - 6) + "╮\n"}
          {"  │" + " ".repeat(width - 6) + "│\n"}
          {"  │" + "   No loss data yet".padEnd(width - 6) + "│\n"}
          {"  │" + " ".repeat(width - 6) + "│\n"}
          {"  ╰" + "─".repeat(width - 6) + "╯"}
        </Text>
      </Box>
    );
  }

  const trainLoss = metrics.map(m => m.loss);
  const valLoss = metrics.filter(m => m.valLoss != null).map(m => m.valLoss!);

  const series = [
    { values: trainLoss, label: "train", char: "●" },
  ];
  if (valLoss.length > 0) {
    series.push({ values: valLoss, label: "val", char: "◦" });
  }

  const lines = renderChart(series, width, height);

  return (
    <Box flexDirection="column">
      <Text color="gray"> {"╶──"} <Text color="white" bold>Loss</Text> {"──╴"}</Text>
      {lines.map((line, i) => {
        // Color the axis gray, the data colored
        if (i < height) {
          // Chart row — color the plot characters
          const parts = line.split("│");
          if (parts.length >= 2) {
            return (
              <Text key={i}>
                <Text color="gray" dimColor>{parts[0]}{"│"}</Text>
                <Text>{colorPlotLine(parts.slice(1).join("│"))}</Text>
              </Text>
            );
          }
          const rParts = line.split("┐");
          if (rParts.length >= 2) {
            return (
              <Text key={i}>
                <Text color="gray" dimColor>{rParts[0]}{"┐"}</Text>
                <Text>{colorPlotLine(rParts.slice(1).join("┐"))}</Text>
              </Text>
            );
          }
          const tParts = line.split("┤");
          if (tParts.length >= 2) {
            return (
              <Text key={i}>
                <Text color="gray" dimColor>{tParts[0]}{"┤"}</Text>
                <Text>{colorPlotLine(tParts.slice(1).join("┤"))}</Text>
              </Text>
            );
          }
        }
        return <Text key={i} color="gray" dimColor>{line}</Text>;
      })}
    </Box>
  );
}

function colorPlotLine(plotPart: string): React.ReactNode {
  // Color ● as yellow, ◦ as cyan, │ as gray dim
  const chars = [...plotPart];
  return chars.map((c, i) => {
    if (c === "●") return <Text key={i} color="yellow" bold>{c}</Text>;
    if (c === "◦") return <Text key={i} color="cyanBright">{c}</Text>;
    if (c === "│") return <Text key={i} color="gray" dimColor>{c}</Text>;
    return <Text key={i}>{c}</Text>;
  });
}
