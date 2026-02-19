import React from "react";
import { Box, Text, useStdout } from "ink";
import { TABS, TAB_LABELS, type Tab } from "../types.js";

export function TabBar({ activeTab }: { activeTab: Tab }) {
  const { stdout } = useStdout();
  const cols = (stdout?.columns ?? 80) - 2;

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box flexDirection="row" gap={0}>
        {TABS.map((tab, i) => {
          const active = tab === activeTab;
          const label = TAB_LABELS[tab];
          const num = String(i + 1);
          return (
            <Box key={tab}>
              {active ? (
                <Text>
                  <Text color="cyanBright" bold> {label} </Text>
                </Text>
              ) : (
                <Text>
                  <Text color="gray" dimColor> </Text>
                  <Text color="gray" dimColor>{num}</Text>
                  <Text color="gray"> {label} </Text>
                </Text>
              )}
            </Box>
          );
        })}
      </Box>
      {/* Underline: bright under active tab, dim elsewhere */}
      <Box flexDirection="row" gap={0}>
        {TABS.map(tab => {
          const active = tab === activeTab;
          const w = TAB_LABELS[tab].length + 2;
          // +1 for the number prefix on inactive tabs
          const totalW = active ? w : w + 2;
          return (
            <Text key={tab} color={active ? "cyan" : "gray"} dimColor={!active}>
              {active ? "━".repeat(totalW) : "─".repeat(totalW)}
            </Text>
          );
        })}
        {/* Fill rest of line */}
        <Text color="gray" dimColor>{"─".repeat(Math.max(0, cols - TABS.reduce((s, t) => s + TAB_LABELS[t].length + (t === activeTab ? 2 : 4), 0)))}</Text>
      </Box>
    </Box>
  );
}
