import React from "react";
import { Box, Text, useStdout } from "ink";
import type { Tab, ViewMode } from "../types.js";

function Key({ k, desc }: { k: string; desc: string }) {
  return (
    <Text>
      <Text color="gray"> </Text>
      <Text color="white" bold backgroundColor="gray">{` ${k} `}</Text>
      <Text color="gray"> {desc} </Text>
    </Text>
  );
}

export function HelpBar({ tab, view }: { tab: Tab; view: ViewMode }) {
  const { stdout } = useStdout();
  const cols = (stdout?.columns ?? 80) - 2;

  return (
    <Box flexDirection="column">
      <Box paddingX={1}>
        <Text color="gray" dimColor>{"─".repeat(cols)}</Text>
      </Box>
      <Box flexDirection="row" paddingX={1}>
        {view === "detail" ? (
          <>
            <Key k="esc" desc="back" />
            <Key k="j/k" desc="scroll" />
            <Key k="q" desc="quit" />
          </>
        ) : (
          <>
            <Key k="1-4" desc="tab" />
            <Key k="j/k" desc="navigate" />
            <Key k="enter" desc="open" />
            {tab === "monitor" && <Key k="r" desc="refresh" />}
            <Key k="q" desc="quit" />
          </>
        )}
      </Box>
    </Box>
  );
}
