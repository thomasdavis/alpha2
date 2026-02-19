import React from "react";
import { Box, Text, useStdout } from "ink";
import type { LogFile } from "../types.js";
import { formatBytes, timeAgo } from "../lib/format.js";

export function LogsView({
  logs,
  selectedIndex,
  selectedLog,
  scrollOffset,
}: {
  logs: LogFile[];
  selectedIndex: number;
  selectedLog: LogFile | null;
  scrollOffset: number;
}) {
  const { stdout } = useStdout();
  const rows = (stdout?.rows ?? 24) - 10;

  if (selectedLog) {
    return <LogDetail log={selectedLog} scrollOffset={scrollOffset} rows={rows} />;
  }

  if (logs.length === 0) {
    return (
      <Box flexDirection="column" paddingX={2} paddingY={1}>
        <Text color="gray" dimColor>No log files found in outputs/</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box flexDirection="row" gap={1}>
        <Text color="gray" dimColor>{"  "}</Text>
        <Box width={30}><Text color="gray">Log File</Text></Box>
        <Box width={10}><Text color="gray">Size</Text></Box>
        <Box width={8}><Text color="gray">Lines</Text></Box>
        <Box width={10}><Text color="gray">Modified</Text></Box>
      </Box>
      {logs.map((log, i) => {
        const selected = i === selectedIndex;
        return (
          <Box key={log.name} flexDirection="row" gap={1}>
            <Text color={selected ? "cyanBright" : undefined}>{selected ? "▸" : " "}</Text>
            <Text color={selected ? "cyanBright" : "gray"}>{"📄"}</Text>
            <Box width={28}>
              <Text bold={selected} color={selected ? "cyanBright" : "white"} wrap="truncate">
                {log.name}
              </Text>
            </Box>
            <Box width={10}>
              <Text color="gray" dimColor>{formatBytes(log.size)}</Text>
            </Box>
            <Box width={8}>
              <Text color="gray" dimColor>{log.lines.length}</Text>
            </Box>
            <Box width={10}>
              <Text color="gray" dimColor>{timeAgo(log.mtime)}</Text>
            </Box>
          </Box>
        );
      })}
    </Box>
  );
}

function LogDetail({
  log,
  scrollOffset,
  rows,
}: {
  log: LogFile;
  scrollOffset: number;
  rows: number;
}) {
  const { stdout } = useStdout();
  const cols = (stdout?.columns ?? 80) - 4;
  const visibleLines = log.lines.slice(scrollOffset, scrollOffset + rows);
  const totalLines = log.lines.length;
  const pct = totalLines > 0 ? Math.min(100, Math.round(((scrollOffset + rows) / totalLines) * 100)) : 100;

  // Scroll bar
  const scrollBarH = Math.max(rows, 1);
  const thumbH = Math.max(1, Math.round((rows / totalLines) * scrollBarH));
  const thumbPos = Math.round((scrollOffset / Math.max(1, totalLines - rows)) * (scrollBarH - thumbH));

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box flexDirection="row" justifyContent="space-between">
        <Text>
          <Text color="gray">{"📄 "}</Text>
          <Text bold color="cyanBright">{log.name}</Text>
        </Text>
        <Text color="gray" dimColor>
          {scrollOffset + 1}–{Math.min(scrollOffset + rows, totalLines)} of {totalLines}  {pct}%
        </Text>
      </Box>
      <Text color="gray" dimColor>{"─".repeat(cols)}</Text>
      {visibleLines.map((line, i) => {
        const lineNum = scrollOffset + i + 1;
        const numStr = String(lineNum).padStart(4);
        const scrollIdx = i;
        const isThumb = scrollIdx >= thumbPos && scrollIdx < thumbPos + thumbH;
        return (
          <Box key={lineNum} flexDirection="row">
            <Text color="gray" dimColor>{numStr} </Text>
            <Text color={colorForLogLine(line)} wrap="truncate">{line || " "}</Text>
            <Box flexGrow={1} />
            <Text color={isThumb ? "cyan" : "gray"} dimColor={!isThumb}>{isThumb ? "┃" : "│"}</Text>
          </Box>
        );
      })}
    </Box>
  );
}

function colorForLogLine(line: string): string {
  if (line.startsWith("──") || line.startsWith("── ")) return "cyan";
  if (line.startsWith("step")) return "white";
  if (line.includes("error") || line.includes("Error") || line.includes("ERROR")) return "redBright";
  if (line.includes("checkpoint saved") || line.includes("checkpoint:")) return "greenBright";
  if (line.includes("val_loss") || line.includes("valLoss")) return "blueBright";
  if (line.includes("training complete") || line.includes("complete")) return "greenBright";
  if (line.startsWith("Implementations") || line.startsWith("Tokenizers:") || line.startsWith("Backends:") || line.startsWith("Optimizers:")) return "yellow";
  if (line.startsWith("run_id:") || line.startsWith("config_hash:") || line.startsWith("params:") || line.startsWith("backend:") || line.startsWith("seed:") || line.startsWith("iters:")) return "cyan";
  return "gray";
}
