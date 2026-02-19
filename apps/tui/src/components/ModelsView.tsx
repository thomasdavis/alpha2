import React from "react";
import { Box, Text, useStdout } from "ink";
import type { RunState } from "../types.js";
import { formatParams, formatLoss, lossColor, timeAgo } from "../lib/format.js";
import { StatusBadge, StatusLabel } from "./StatusBadge.js";
import { Sparkline } from "./Sparkline.js";

export function ModelsView({
  runs,
  selectedIndex,
  selectedRun,
}: {
  runs: RunState[];
  selectedIndex: number;
  selectedRun: RunState | null;
}) {
  if (selectedRun) {
    return <ModelDetail run={selectedRun} />;
  }

  if (runs.length === 0) {
    return (
      <Box flexDirection="column" paddingX={2} paddingY={1}>
        <Text color="gray" dimColor>No models found.</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box flexDirection="row" gap={1}>
        <Text color="gray" dimColor>{"  "}</Text>
        <Box width={2}><Text> </Text></Box>
        <Box width={17}><Text color="gray">Model</Text></Box>
        <Box width={8}><Text color="gray">Params</Text></Box>
        <Box width={15}><Text color="gray">Architecture</Text></Box>
        <Box width={6}><Text color="gray">Vocab</Text></Box>
        <Box width={9}><Text color="gray">Best Val</Text></Box>
        <Box width={10}><Text color="gray">Tokenizer</Text></Box>
      </Box>
      {runs.map((run, i) => {
        const selected = i === selectedIndex;
        const mc = run.config.modelConfig;
        const arch = `${mc.nLayer}L ${mc.nEmbd}D ${mc.nHead}H`;
        return (
          <Box key={run.name} flexDirection="row" gap={1}>
            <Text color={selected ? "cyanBright" : undefined}>{selected ? "▸" : " "}</Text>
            <Box width={2}><StatusBadge status={run.status} /></Box>
            <Box width={17}>
              <Text bold={selected} color={selected ? "cyanBright" : "white"} wrap="truncate">
                {run.name}
              </Text>
            </Box>
            <Box width={8}><Text color="yellow">{formatParams(run.estimatedParams)}</Text></Box>
            <Box width={15}><Text color="magenta">{arch}</Text></Box>
            <Box width={6}><Text color="gray">{mc.vocabSize}</Text></Box>
            <Box width={9}><Text color={lossColor(run.bestValLoss)}>{formatLoss(run.bestValLoss)}</Text></Box>
            <Box width={10}><Text color="gray">{run.config.trainConfig.tokenizer}</Text></Box>
          </Box>
        );
      })}
    </Box>
  );
}

function ModelDetail({ run }: { run: RunState }) {
  const { stdout } = useStdout();
  const cols = stdout?.columns ?? 80;
  const mc = run.config.modelConfig;
  const tc = run.config.trainConfig;
  const panelW = 28;

  return (
    <Box flexDirection="column" paddingX={1}>
      {/* Title */}
      <Box flexDirection="row" gap={1} marginBottom={1}>
        <StatusBadge status={run.status} />
        <Text bold color="cyanBright">{run.name}</Text>
        <Text color="gray">{"·"}</Text>
        <Text color="magenta">{run.domain}</Text>
        <Text color="gray">{"·"}</Text>
        <Text color="yellow" bold>{formatParams(run.estimatedParams)}</Text>
      </Box>

      <Box flexDirection="row" gap={1}>
        {/* Architecture */}
        <Box flexDirection="column">
          <Text color="gray">{"╭─ "}<Text color="white" bold>Architecture</Text>{" " + "─".repeat(panelW - 17)}{"╮"}</Text>
          <PanelRow label="Vocab Size" value={String(mc.vocabSize)} w={panelW} />
          <PanelRow label="Block Size" value={String(mc.blockSize)} w={panelW} />
          <PanelRow label="Layers" value={String(mc.nLayer)} w={panelW} />
          <PanelRow label="Embed Dim" value={String(mc.nEmbd)} w={panelW} />
          <PanelRow label="Heads" value={String(mc.nHead)} w={panelW} />
          <PanelRow label="Head Dim" value={String(Math.floor(mc.nEmbd / mc.nHead))} w={panelW} />
          <PanelRow label="FFN Dim" value={String(mc.nEmbd * 4)} w={panelW} />
          <PanelRow label="Dropout" value={String(mc.dropout)} w={panelW} />
          <Text color="gray">{"╰" + "─".repeat(panelW - 2) + "╯"}</Text>
        </Box>

        {/* Training */}
        <Box flexDirection="column">
          <Text color="gray">{"╭─ "}<Text color="white" bold>Training</Text>{" " + "─".repeat(panelW - 13)}{"╮"}</Text>
          <PanelRow label="Tokenizer" value={tc.tokenizer} w={panelW} />
          <PanelRow label="Optimizer" value={tc.optimizer} w={panelW} />
          <PanelRow label="LR" value={tc.lr.toExponential(1)} w={panelW} />
          <PanelRow label="Batch Size" value={String(tc.batchSize)} w={panelW} />
          <PanelRow label="Wt Decay" value={String(tc.weightDecay)} w={panelW} />
          <PanelRow label="Grad Clip" value={String(tc.gradClip)} w={panelW} />
          <PanelRow label="Backend" value={tc.backend} w={panelW} />
          <PanelRow label="Seed" value={String(tc.seed)} w={panelW} />
          <Text color="gray">{"╰" + "─".repeat(panelW - 2) + "╯"}</Text>
        </Box>

        {/* Results */}
        <Box flexDirection="column">
          <Text color="gray">{"╭─ "}<Text color="white" bold>Results</Text>{" " + "─".repeat(panelW - 12)}{"╮"}</Text>
          <PanelRow label="Step" value={`${run.latestStep}/${run.totalIters}`} w={panelW} />
          <PanelRow label="Train Loss" value={formatLoss(run.lastLoss)} color={lossColor(run.lastLoss)} w={panelW} />
          <PanelRow label="Best Val" value={formatLoss(run.bestValLoss)} color={lossColor(run.bestValLoss)} w={panelW} />
          <PanelRow label="Avg Tok/s" value={run.avgTokensPerSec.toFixed(0)} w={panelW} />
          <PanelRow label="Checkpoints" value={String(run.checkpoints.length)} w={panelW} />
          <PanelRow label="Hash" value={run.config.configHash} w={panelW} />
          <PanelRow label="Run ID" value={run.config.runId} w={panelW} />
          <PanelRow label="Age" value={timeAgo(run.mtime)} w={panelW} />
          <Text color="gray">{"╰" + "─".repeat(panelW - 2) + "╯"}</Text>
        </Box>
      </Box>

      {/* Parameter breakdown */}
      <Box flexDirection="column" marginTop={1}>
        <Text color="gray"> {"╶──"} <Text color="white" bold>Parameters</Text> {"──" + "─".repeat(Math.max(0, cols - 22))}</Text>
        <ParamBreakdown mc={mc} width={Math.min(cols - 4, 70)} />
      </Box>

      {/* Loss preview */}
      {run.metrics.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text color="gray"> {"╶──"} <Text color="white" bold>Loss Curve</Text> {"──" + "─".repeat(Math.max(0, cols - 22))}</Text>
          <Box paddingX={2}>
            <Text color="gray">train </Text>
            <Sparkline values={run.metrics.map(m => m.loss)} width={Math.min(60, cols - 20)} gradient />
          </Box>
          {run.metrics.some(m => m.valLoss != null) && (
            <Box paddingX={2}>
              <Text color="gray">val   </Text>
              <Sparkline values={run.metrics.filter(m => m.valLoss != null).map(m => m.valLoss!)} width={Math.min(60, cols - 20)} gradient />
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}

function PanelRow({ label, value, color, w }: { label: string; value: string; color?: string; w: number }) {
  const inner = w - 4;
  const content = `${label.padEnd(12)}${value}`;
  const trailing = Math.max(0, inner - content.length);
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

function ParamBreakdown({ mc, width }: { mc: RunState["config"]["modelConfig"]; width: number }) {
  const E = mc.nEmbd, L = mc.nLayer, V = mc.vocabSize, B = mc.blockSize;

  const tokEmbed = V * E;
  const posEmbed = B * E;
  const attnPerLayer = 4 * E * E + 4 * E;
  const ffnPerLayer = 4 * 4 * E * E + 4 * E;
  const lnPerLayer = 2 * E;
  const finalLn = 2 * E;
  const lmHead = V * E;
  const total = tokEmbed + posEmbed + L * (attnPerLayer + ffnPerLayer + lnPerLayer) + finalLn + lmHead;

  const barW = Math.max(10, width - 28);
  const fmt = (n: number) => formatParams(n);
  const bar = (n: number) => {
    const pct = n / total;
    const w = Math.max(1, Math.round(pct * barW));
    return "█".repeat(w) + "░".repeat(Math.max(0, barW - w));
  };

  const items = [
    { label: "Token Embed", n: tokEmbed, color: "cyan" },
    { label: "Pos Embed", n: posEmbed, color: "blue" },
    { label: `Attn x${L}`, n: attnPerLayer * L, color: "yellow" },
    { label: `FFN x${L}`, n: ffnPerLayer * L, color: "green" },
    { label: `LN x${L}`, n: lnPerLayer * L, color: "gray" },
    { label: "Final LN", n: finalLn, color: "gray" },
    { label: "LM Head", n: lmHead, color: "magenta" },
  ];

  return (
    <Box flexDirection="column" paddingX={2}>
      {items.map(({ label, n, color }) => (
        <Box key={label} flexDirection="row">
          <Text color="gray" dimColor>{label.padEnd(14)}</Text>
          <Text color="gray">{fmt(n).padStart(6)} </Text>
          <Text color={color}>{bar(n)}</Text>
        </Box>
      ))}
      <Box flexDirection="row" marginTop={0}>
        <Text color="gray" dimColor>{"─".repeat(14)}</Text>
        <Text color="gray" dimColor>{"─".repeat(7)}</Text>
      </Box>
      <Box flexDirection="row">
        <Text color="gray">{"Total".padEnd(14)}</Text>
        <Text color="yellowBright" bold>{fmt(total).padStart(6)}</Text>
      </Box>
    </Box>
  );
}
