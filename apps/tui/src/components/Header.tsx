import React from "react";
import { Box, Text, useStdout } from "ink";
import type { RunState, Tab } from "../types.js";
import { TabBar } from "./TabBar.js";

export function Header({ runs, activeTab }: { runs: RunState[]; activeTab: Tab }) {
  const { stdout } = useStdout();
  const cols = (stdout?.columns ?? 80) - 2;
  const active = runs.filter(r => r.status === "active").length;
  const completed = runs.filter(r => r.status === "completed").length;
  const total = runs.length;

  return (
    <Box flexDirection="column">
      {/* Top border */}
      <Box paddingX={0}>
        <Text color="gray"> {"╭" + "─".repeat(cols - 2) + "╮"}</Text>
      </Box>

      {/* Title bar */}
      <Box flexDirection="row" justifyContent="space-between">
        <Text>
          <Text color="gray"> │ </Text>
          <Text color="cyan" bold>{"▲"}</Text>
          <Text color="cyanBright" bold> ALPHA</Text>
          <Text color="gray"> training monitor</Text>
        </Text>
        <Text>
          <Text color="white" bold>{total}</Text>
          <Text color="gray"> runs</Text>
          {active > 0 && (
            <Text>
              <Text color="gray"> {"·"} </Text>
              <Text color="greenBright" bold>{active}</Text>
              <Text color="green"> active</Text>
            </Text>
          )}
          {completed > 0 && (
            <Text>
              <Text color="gray"> {"·"} </Text>
              <Text color="blue">{completed}</Text>
              <Text color="gray"> done</Text>
            </Text>
          )}
          <Text color="gray"> │</Text>
        </Text>
      </Box>

      {/* Bottom border + tabs */}
      <Box paddingX={0}>
        <Text color="gray"> {"╰" + "─".repeat(cols - 2) + "╯"}</Text>
      </Box>

      <TabBar activeTab={activeTab} />
    </Box>
  );
}
