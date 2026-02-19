import React from "react";
import { Box, Text, useStdout } from "ink";
import type { RunState } from "../types.js";
import { RunRow } from "./RunRow.js";
import { Sparkline } from "./Sparkline.js";
import { formatLoss, lossColor, timeAgo } from "../lib/format.js";

export function Dashboard({
  runs,
  selectedIndex,
}: {
  runs: RunState[];
  selectedIndex: number;
}) {
  const { stdout } = useStdout();
  const cols = Math.max((stdout?.columns ?? 80) - 4, 60);

  if (runs.length === 0) {
    return (
      <Box flexDirection="column" paddingX={2} paddingY={1}>
        <Text color="gray">
          {"\n"}
          {"  ╭───────────────────────────────────╮\n"}
          {"  │                                   │\n"}
          {"  │   No training runs found.         │\n"}
          {"  │                                   │\n"}
          {"  │   Start a run to see it here:     │\n"}
          {"  │   npx tsx apps/cli/src/main.ts    │\n"}
          {"  │                                   │\n"}
          {"  ╰───────────────────────────────────╯"}
        </Text>
      </Box>
    );
  }

  const selected = runs[selectedIndex];

  return (
    <Box flexDirection="column" paddingX={1}>
      {/* Column headers */}
      <Box flexDirection="row">
        <Text color="gray" dimColor>{"    "}</Text>
        <Box width={16}><Text color="gray" dimColor>Name</Text></Box>
        <Box width={9}><Text color="gray" dimColor>Domain</Text></Box>
        <Box width={7}><Text color="gray" dimColor>Params</Text></Box>
        <Box width={16}><Text color="gray" dimColor>Progress</Text></Box>
        <Box width={8}><Text color="gray" dimColor>Loss</Text></Box>
        <Box width={8}><Text color="gray" dimColor>Val</Text></Box>
        <Box width={5}><Text color="gray" dimColor>Tok/s</Text></Box>
      </Box>

      {/* Runs */}
      {runs.map((run, i) => (
        <RunRow key={run.name} run={run} selected={i === selectedIndex} />
      ))}

      {/* Selected run preview */}
      {selected && selected.metrics.length > 0 && (
        <Box flexDirection="column" marginTop={1} paddingX={1}>
          <Text color="gray" dimColor>{"╶" + "─".repeat(cols - 2) + "╴"}</Text>
          <Box flexDirection="row" gap={0} marginTop={0}>
            <Text color="gray" dimColor> loss  </Text>
            <Sparkline
              values={selected.metrics.map(m => m.loss)}
              width={Math.min(45, cols - 35)}
              gradient
            />
            <Text color="gray" dimColor>  </Text>
            <Text color={lossColor(selected.lastLoss)}>{formatLoss(selected.lastLoss)}</Text>
            {selected.bestValLoss != null && (
              <Text>
                <Text color="gray" dimColor>  val </Text>
                <Text color={lossColor(selected.bestValLoss)}>{formatLoss(selected.bestValLoss)}</Text>
              </Text>
            )}
            <Text color="gray" dimColor>  {timeAgo(selected.mtime)}</Text>
          </Box>
          {selected.metrics.some(m => m.valLoss != null) && (
            <Box flexDirection="row" gap={0}>
              <Text color="gray" dimColor> val   </Text>
              <Sparkline
                values={selected.metrics.filter(m => m.valLoss != null).map(m => m.valLoss!)}
                width={Math.min(45, cols - 35)}
                gradient
              />
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}
