import React from "react";
import { Box, Text, useStdout } from "ink";
import type { RunState } from "../types.js";
import { LossChart } from "./LossChart.js";
import { StatusBadge, StatusLabel } from "./StatusBadge.js";
import { Sparkline } from "./Sparkline.js";
import { formatParams, formatLoss, formatEta, formatTokPerSec, progressBar, lossColor, timeAgo } from "../lib/format.js";

function Panel({ title, width, children }: { title: string; width: number; children: React.ReactNode }) {
  const inner = width - 4;
  return (
    <Box flexDirection="column">
      <Text color="gray">{"╭─ "}<Text color="white" bold>{title}</Text>{" " + "─".repeat(Math.max(0, inner - title.length - 2))}{"╮"}</Text>
      <Box flexDirection="column">
        {children}
      </Box>
      <Text color="gray">{"╰" + "─".repeat(width - 2) + "╯"}</Text>
    </Box>
  );
}

function PanelRow({ label, value, color, width }: { label: string; value: string; color?: string; width: number }) {
  const inner = width - 4;
  const padded = `${label.padEnd(12)}${value}`;
  const trailing = Math.max(0, inner - padded.length);
  return (
    <Text>
      <Text color="gray">{"│ "}</Text>
      <Text color="gray" dimColor>{label.padEnd(12)}</Text>
      <Text color={color ?? "white"}>{value}</Text>
      <Text>{" ".repeat(trailing)}</Text>
      <Text color="gray">{" │"}</Text>
    </Text>
  );
}

export function DetailView({
  run,
  scrollOffset,
}: {
  run: RunState;
  scrollOffset: number;
}) {
  const { stdout } = useStdout();
  const cols = stdout?.columns ?? 80;
  const chartWidth = Math.min(cols - 32, 55);
  const panelW = 28;
  const mc = run.config.modelConfig;
  const tc = run.config.trainConfig;
  const pb = progressBar(run.latestStep, run.totalIters, 16);
  const done = run.latestStep >= run.totalIters;

  const recentMetrics = run.metrics.slice(-20);
  const displayMetrics = recentMetrics.slice(scrollOffset);

  return (
    <Box flexDirection="column" paddingX={1}>
      {/* Title bar */}
      <Box flexDirection="row" gap={1} marginBottom={0}>
        <StatusBadge status={run.status} />
        <Text bold color="cyanBright">{run.name}</Text>
        <Text color="gray">{"·"}</Text>
        <StatusLabel status={run.status} />
        <Text color="gray">{"·"}</Text>
        <Text color="magenta">{run.domain}</Text>
        <Text color="gray">{"·"}</Text>
        <Text color="yellow" bold>{formatParams(run.estimatedParams)}</Text>
        <Text color="gray">{"·"}</Text>
        <Text color="gray" dimColor>{timeAgo(run.mtime)}</Text>
      </Box>

      {/* Progress */}
      <Box flexDirection="row" gap={1} marginBottom={1}>
        <Text color="gray">  </Text>
        <Text color={done ? "greenBright" : "cyan"}>{pb.filled}</Text>
        <Text color="gray" dimColor>{pb.empty}</Text>
        <Text color="gray"> {run.latestStep}/{run.totalIters}</Text>
        {run.etaMs != null && (
          <Text color="gray" dimColor> {"·"} ETA {formatEta(run.etaMs)}</Text>
        )}
      </Box>

      <Box flexDirection="row" gap={1}>
        {/* Config panel */}
        <Panel title="Config" width={panelW}>
          <PanelRow label="Tokenizer" value={tc.tokenizer} width={panelW} />
          <PanelRow label="Vocab" value={String(mc.vocabSize)} width={panelW} />
          <PanelRow label="Block Size" value={String(mc.blockSize)} width={panelW} />
          <PanelRow label="Layers" value={String(mc.nLayer)} width={panelW} />
          <PanelRow label="Embed Dim" value={String(mc.nEmbd)} width={panelW} />
          <PanelRow label="Heads" value={String(mc.nHead)} width={panelW} />
          <PanelRow label="Batch" value={String(tc.batchSize)} width={panelW} />
          <PanelRow label="LR" value={tc.lr.toExponential(1)} width={panelW} />
          <PanelRow label="Optimizer" value={tc.optimizer} width={panelW} />
          <PanelRow label="Loss" value={formatLoss(run.lastLoss)} color={lossColor(run.lastLoss)} width={panelW} />
          <PanelRow label="Best Val" value={formatLoss(run.bestValLoss)} color={lossColor(run.bestValLoss)} width={panelW} />
          <PanelRow label="Tok/s" value={formatTokPerSec(run.avgTokensPerSec)} width={panelW} />
          {run.checkpoints.length > 0 && (
            <PanelRow label="Checkpoints" value={String(run.checkpoints.length)} width={panelW} />
          )}
        </Panel>

        {/* Loss chart */}
        <Box flexDirection="column">
          <LossChart metrics={run.metrics} width={chartWidth} height={9} />
        </Box>
      </Box>

      {/* Recent metrics table */}
      {displayMetrics.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text color="gray"> {"╶──"} <Text color="white" bold>Recent Steps</Text> {"──" + "─".repeat(Math.max(0, cols - 25))}</Text>
          <Box flexDirection="row" gap={1} paddingX={1}>
            <Box width={7}><Text color="gray">Step</Text></Box>
            <Box width={9}><Text color="gray">Loss</Text></Box>
            <Box width={9}><Text color="gray">Val Loss</Text></Box>
            <Box width={10}><Text color="gray">LR</Text></Box>
            <Box width={10}><Text color="gray">Grad</Text></Box>
            <Box width={9}><Text color="gray">ms/iter</Text></Box>
            <Box width={8}><Text color="gray">Tok/s</Text></Box>
          </Box>
          {displayMetrics.map(m => (
            <Box key={m.step} flexDirection="row" gap={1} paddingX={1}>
              <Box width={7}><Text color="white" dimColor>{m.step}</Text></Box>
              <Box width={9}><Text color={lossColor(m.loss)}>{formatLoss(m.loss)}</Text></Box>
              <Box width={9}><Text color={m.valLoss != null ? lossColor(m.valLoss) : "gray"}>{formatLoss(m.valLoss)}</Text></Box>
              <Box width={10}><Text color="gray" dimColor>{m.lr.toExponential(1)}</Text></Box>
              <Box width={10}><Text color={m.gradNorm > 100 ? "yellow" : "gray"} dimColor>{m.gradNorm.toFixed(1)}</Text></Box>
              <Box width={9}><Text color="gray" dimColor>{m.ms_per_iter.toFixed(0)}</Text></Box>
              <Box width={8}><Text color="gray" dimColor>{formatTokPerSec(m.tokens_per_sec)}</Text></Box>
            </Box>
          ))}
        </Box>
      )}
    </Box>
  );
}
