import React from "react";
import { Box, Text } from "ink";
import type { RunState } from "../types.js";
import { StatusBadge } from "./StatusBadge.js";
import { formatParams, formatLoss, formatTokPerSec, progressBar, lossColor } from "../lib/format.js";

export function RunRow({
  run,
  selected,
}: {
  run: RunState;
  selected: boolean;
}) {
  const pointer = selected ? "▸" : " ";
  const done = run.latestStep >= run.totalIters;
  const pb = progressBar(run.latestStep, run.totalIters, 8);

  return (
    <Box flexDirection="row">
      <Text color={selected ? "cyanBright" : undefined}>{pointer}</Text>
      <Text> </Text>
      <Box width={2}>
        <StatusBadge status={run.status} />
      </Box>
      <Text> </Text>
      <Box width={16}>
        <Text bold={selected} color={selected ? "cyanBright" : "white"} wrap="truncate">
          {run.name}
        </Text>
      </Box>
      <Box width={9}>
        <Text color="magenta">{run.domain}</Text>
      </Box>
      <Box width={7}>
        <Text color="yellow">{formatParams(run.estimatedParams)}</Text>
      </Box>
      <Box width={16}>
        <Text>
          <Text color={done ? "greenBright" : "cyan"}>{pb.filled}</Text>
          <Text color="gray" dimColor>{pb.empty}</Text>
          <Text color={done ? "greenBright" : "gray"} dimColor={!done}>{done ? " done" : ` ${pb.pct}%`}</Text>
        </Text>
      </Box>
      <Box width={8}>
        <Text color={lossColor(run.lastLoss)}>{formatLoss(run.lastLoss)}</Text>
      </Box>
      <Box width={8}>
        <Text color={lossColor(run.bestValLoss)}>{formatLoss(run.bestValLoss)}</Text>
      </Box>
      <Box width={5}>
        <Text color="gray" dimColor>{formatTokPerSec(run.avgTokensPerSec)}</Text>
      </Box>
    </Box>
  );
}
