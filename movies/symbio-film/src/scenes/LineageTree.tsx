import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Easing,
  Sequence,
} from "remotion";
import { THEME } from "../theme";
import {
  fontFamily,
  SceneBackground,
  SceneTitle,
  FadeIn,
  GlowDot,
} from "../components/shared";
import { metrics, getActivationColor, extractSwitchEvents } from "../data";

// Build a tree structure from candidate lineage data
type TreeNode = {
  name: string;
  activation: string;
  generation: number;
  parent: string | null;
  bestLoss: number;
  steps: number;
  children: TreeNode[];
  color: string;
  x: number;
  y: number;
};

function buildTree(): TreeNode[] {
  const candidateMap = new Map<
    string,
    {
      name: string;
      activation: string;
      generation: number;
      parent: string | null;
      losses: number[];
    }
  >();

  for (const m of metrics) {
    const name = m.symbio_candidate_name;
    if (!name) continue;
    const entry = candidateMap.get(name) || {
      name,
      activation: m.symbio_candidate_activation || "",
      generation: m.symbio_generation || 0,
      parent: m.symbio_candidate_parent_name || null,
      losses: [],
    };
    entry.losses.push(m.loss);
    candidateMap.set(name, entry);
  }

  const nodes: TreeNode[] = [];
  for (const [, c] of candidateMap) {
    nodes.push({
      name: c.name,
      activation: c.activation,
      generation: c.generation,
      parent: c.parent,
      bestLoss: Math.min(...c.losses),
      steps: c.losses.length,
      children: [],
      color: getActivationColor(c.activation),
      x: 0,
      y: 0,
    });
  }

  // Build parent-child links
  const byName = new Map<string, TreeNode>();
  for (const n of nodes) byName.set(n.name, n);

  const roots: TreeNode[] = [];
  for (const n of nodes) {
    if (n.parent && byName.has(n.parent)) {
      byName.get(n.parent)!.children.push(n);
    } else {
      roots.push(n);
    }
  }

  return roots;
}

// Layout tree nodes with x,y coordinates
function layoutTree(
  roots: TreeNode[],
  width: number,
  height: number
): TreeNode[] {
  const allNodes: TreeNode[] = [];
  const maxGen = Math.max(
    ...getAllNodes(roots).map((n) => n.generation),
    0
  );
  const genHeight = height / (maxGen + 1);

  function getAllNodes(nodes: TreeNode[]): TreeNode[] {
    const result: TreeNode[] = [];
    for (const n of nodes) {
      result.push(n);
      result.push(...getAllNodes(n.children));
    }
    return result;
  }

  // Group by generation for x positioning
  const byGen = new Map<number, TreeNode[]>();
  for (const n of getAllNodes(roots)) {
    const list = byGen.get(n.generation) || [];
    list.push(n);
    byGen.set(n.generation, list);
  }

  for (const [gen, nodes] of byGen) {
    const y = 80 + gen * genHeight;
    const spacing = width / (nodes.length + 1);
    nodes.forEach((n, i) => {
      n.x = spacing * (i + 1);
      n.y = y;
      allNodes.push(n);
    });
  }

  return allNodes;
}

// Animated Bezier link between parent and child
const TreeLink: React.FC<{
  parent: TreeNode;
  child: TreeNode;
  progress: number;
}> = ({ parent, child, progress }) => {
  const midY = (parent.y + child.y) / 2;
  const pathD = `M ${parent.x} ${parent.y} C ${parent.x} ${midY}, ${child.x} ${midY}, ${child.x} ${child.y}`;

  const drawProgress = Math.min(1, progress);
  const totalLength = 500; // approximate

  return (
    <path
      d={pathD}
      fill="none"
      stroke={child.color}
      strokeWidth={2}
      opacity={drawProgress * 0.5}
      strokeDasharray={totalLength}
      strokeDashoffset={totalLength * (1 - drawProgress)}
      strokeLinecap="round"
    />
  );
};

// Tree node visual
const TreeNodeViz: React.FC<{
  node: TreeNode;
  progress: number;
  isGlobalBest: boolean;
}> = ({ node, progress, isGlobalBest }) => {
  // Size by quality (lower loss = bigger)
  const radius = interpolate(node.bestLoss, [6.0, 8.5], [28, 10], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const scale = progress;

  return (
    <g transform={`translate(${node.x}, ${node.y})`}>
      {/* Glow for best */}
      {isGlobalBest && (
        <>
          <circle
            r={radius + 16}
            fill="none"
            stroke={THEME.amber}
            strokeWidth={2}
            opacity={0.6 * progress}
            strokeDasharray="6,3"
          />
          <circle
            r={radius + 8}
            fill={`${THEME.amber}10`}
            opacity={progress}
          />
        </>
      )}
      {/* Main node */}
      <circle
        r={radius * scale}
        fill={node.color}
        opacity={0.85}
        style={{ filter: `drop-shadow(0 0 10px ${node.color}50)` }}
      />
      {/* Inner highlight */}
      <circle
        r={radius * 0.35 * scale}
        fill="white"
        opacity={0.25 * progress}
      />
      {/* Loss label */}
      <text
        y={-radius - 8}
        fill={THEME.text}
        fontSize={12}
        fontWeight={700}
        textAnchor="middle"
        fontFamily={fontFamily}
        opacity={progress}
      >
        {node.bestLoss.toFixed(3)}
      </text>
      {/* Activation label */}
      <text
        y={radius + 16}
        fill={THEME.textMuted}
        fontSize={10}
        textAnchor="middle"
        fontFamily={fontFamily}
        opacity={progress}
      >
        {node.activation.length > 18
          ? node.activation.slice(0, 15) + "..."
          : node.activation}
      </text>
      {/* Steps label */}
      <text
        y={radius + 30}
        fill={THEME.textDim}
        fontSize={9}
        textAnchor="middle"
        fontFamily={fontFamily}
        opacity={progress * 0.7}
      >
        {node.steps} steps
      </text>
    </g>
  );
};

// Force-directed-style circular layout for the second view
const ForceGraph: React.FC<{
  delay: number;
}> = ({ delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const candidateMap = new Map<
    string,
    {
      name: string;
      activation: string;
      generation: number;
      parent: string | null;
      bestLoss: number;
      steps: number;
      color: string;
    }
  >();

  for (const m of metrics) {
    const name = m.symbio_candidate_name;
    if (!name) continue;
    const entry = candidateMap.get(name) || {
      name,
      activation: m.symbio_candidate_activation || "",
      generation: m.symbio_generation || 0,
      parent: m.symbio_candidate_parent_name || null,
      bestLoss: Infinity,
      steps: 0,
      color: getActivationColor(m.symbio_candidate_activation),
    };
    entry.bestLoss = Math.min(entry.bestLoss, m.loss);
    entry.steps += 1;
    candidateMap.set(name, entry);
  }

  const candidates = Array.from(candidateMap.values());
  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200 },
    durationInFrames: 60,
  });

  // Arrange in concentric rings by generation
  const maxGen = Math.max(...candidates.map((c) => c.generation), 0);
  const centerX = 960;
  const centerY = 500;

  const positioned = candidates.map((c) => {
    const genCandidates = candidates.filter(
      (cc) => cc.generation === c.generation
    );
    const idxInGen = genCandidates.indexOf(c);
    const angleSpan = (Math.PI * 2) / Math.max(genCandidates.length, 1);
    const angle = idxInGen * angleSpan - Math.PI / 2;
    const radius = 100 + c.generation * 50;

    return {
      ...c,
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    };
  });

  const byName = new Map(positioned.map((p) => [p.name, p]));

  return (
    <svg
      width={1920}
      height={1080}
      style={{ position: "absolute", top: 0, left: 0 }}
    >
      {/* Generation rings */}
      {Array.from({ length: maxGen + 1 }, (_, g) => (
        <circle
          key={g}
          cx={centerX}
          cy={centerY}
          r={100 + g * 50}
          fill="none"
          stroke={THEME.gridLine}
          strokeWidth={0.5}
          opacity={progress * 0.3}
        />
      ))}

      {/* Links */}
      {positioned.map((p) => {
        if (!p.parent || !byName.has(p.parent)) return null;
        const parent = byName.get(p.parent)!;
        return (
          <line
            key={`link-${p.name}`}
            x1={parent.x}
            y1={parent.y}
            x2={p.x}
            y2={p.y}
            stroke={p.color}
            strokeWidth={1.5}
            opacity={progress * 0.3}
            strokeLinecap="round"
          />
        );
      })}

      {/* Nodes */}
      {positioned.map((p, i) => {
        const nodeProgress = spring({
          frame: frame - delay - i * 3,
          fps,
          config: { damping: 18, stiffness: 100 },
        });

        const r = interpolate(p.bestLoss, [6.0, 8.5], [22, 6], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });

        return (
          <g key={p.name}>
            <circle
              cx={p.x}
              cy={p.y}
              r={r * nodeProgress}
              fill={p.color}
              opacity={0.8}
              style={{
                filter: `drop-shadow(0 0 6px ${p.color}50)`,
              }}
            />
            <text
              x={p.x}
              y={p.y + r + 14}
              fill={THEME.textMuted}
              fontSize={9}
              textAnchor="middle"
              fontFamily={fontFamily}
              opacity={nodeProgress}
            >
              {p.activation.length > 14
                ? p.activation.slice(0, 11) + "..."
                : p.activation}
            </text>
          </g>
        );
      })}

      {/* Center label */}
      <text
        x={centerX}
        y={centerY - 15}
        fill={THEME.text}
        fontSize={20}
        fontWeight={700}
        textAnchor="middle"
        fontFamily={fontFamily}
        opacity={progress}
      >
        Candidate
      </text>
      <text
        x={centerX}
        y={centerY + 10}
        fill={THEME.accent}
        fontSize={20}
        fontWeight={700}
        textAnchor="middle"
        fontFamily={fontFamily}
        opacity={progress}
      >
        Lineage Graph
      </text>
    </svg>
  );
};

export const LineageTreeScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const roots = buildTree();
  const treeWidth = 1700;
  const treeHeight = 800;

  function getAllNodesFromRoots(nodes: TreeNode[]): TreeNode[] {
    const result: TreeNode[] = [];
    for (const n of nodes) {
      result.push(n);
      result.push(...getAllNodesFromRoots(n.children));
    }
    return result;
  }

  const allNodes = layoutTree(roots, treeWidth, treeHeight);
  const allFlat = getAllNodesFromRoots(roots);
  const globalBest = allFlat.reduce(
    (best, n) => (n.bestLoss < best.bestLoss ? n : best),
    allFlat[0]
  );

  // Phase 1: Tree diagram (first half)
  const treePhaseDuration = Math.floor(durationInFrames * 0.55);
  const inTreePhase = frame < treePhaseDuration;

  // Phase 2: Force graph (second half)
  const forcePhaseStart = treePhaseDuration;

  const fadeOut = interpolate(
    frame,
    [durationInFrames - 30, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Generation band labels
  const maxGen = Math.max(...allFlat.map((n) => n.generation), 0);
  const genHeight = treeHeight / (maxGen + 1);

  return (
    <SceneBackground variant="dark">
      <AbsoluteFill style={{ opacity: fadeOut }}>
        {inTreePhase ? (
          // Tree diagram view
          <>
            <div style={{ padding: "40px 80px" }}>
              <SceneTitle
                title="Candidate Lineage Tree"
                subtitle={`${allFlat.length} candidates across ${maxGen + 1} generations — node size reflects loss quality`}
              />
            </div>

            <svg
              width={1920}
              height={1080}
              style={{ position: "absolute", top: 0, left: 0 }}
            >
              <g transform="translate(110, 100)">
                {/* Generation band labels */}
                {Array.from({ length: maxGen + 1 }, (_, g) => {
                  const y = 80 + g * genHeight;
                  const bandProgress = spring({
                    frame: frame - g * 10,
                    fps,
                    config: { damping: 200 },
                  });
                  return (
                    <g key={g} opacity={bandProgress * 0.5}>
                      <line
                        x1={0}
                        y1={y}
                        x2={treeWidth}
                        y2={y}
                        stroke={THEME.gridLine}
                        strokeWidth={0.5}
                        strokeDasharray="4,8"
                      />
                      <text
                        x={-10}
                        y={y + 4}
                        fill={THEME.textDim}
                        fontSize={12}
                        textAnchor="end"
                        fontFamily={fontFamily}
                      >
                        Gen {g}
                      </text>
                    </g>
                  );
                })}

                {/* Links */}
                {allFlat.map((node) =>
                  node.children.map((child) => {
                    const linkDelay =
                      30 + child.generation * 15;
                    const linkProgress = spring({
                      frame: frame - linkDelay,
                      fps,
                      config: { damping: 200 },
                    });
                    return (
                      <TreeLink
                        key={`${node.name}-${child.name}`}
                        parent={node}
                        child={child}
                        progress={linkProgress}
                      />
                    );
                  })
                )}

                {/* Nodes */}
                {allNodes.map((node, i) => {
                  const nodeDelay = 20 + node.generation * 15 + i * 2;
                  const nodeProgress = spring({
                    frame: frame - nodeDelay,
                    fps,
                    config: { damping: 18, stiffness: 100 },
                  });
                  return (
                    <TreeNodeViz
                      key={node.name}
                      node={node}
                      progress={nodeProgress}
                      isGlobalBest={node.name === globalBest?.name}
                    />
                  );
                })}
              </g>
            </svg>
          </>
        ) : (
          // Force graph view
          <>
            <div style={{ padding: "40px 80px" }}>
              <SceneTitle
                title="Evolutionary Force Graph"
                subtitle="Concentric rings by generation — edges show parent-child mutation paths"
                delay={0}
              />
            </div>
            <ForceGraph delay={10} />
          </>
        )}

        {/* Legend */}
        <Sequence from={40} layout="none">
          <FadeIn delay={0}>
            <div
              style={{
                position: "absolute",
                bottom: 40,
                left: 80,
                display: "flex",
                gap: 20,
              }}
            >
              {["silu", "relu", "gelu", "id", "sq"].map((act) => (
                <div
                  key={act}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: 13,
                    fontFamily,
                  }}
                >
                  <GlowDot color={getActivationColor(act)} size={8} />
                  <span style={{ color: THEME.textMuted }}>{act}</span>
                </div>
              ))}
            </div>
          </FadeIn>
        </Sequence>
      </AbsoluteFill>
    </SceneBackground>
  );
};
