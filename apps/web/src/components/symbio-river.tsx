"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls, Line, Text } from "@react-three/drei";
import * as THREE from "three";
import { type ChartMetric, ChartPanel, fmtNum } from "@/components/charts";

// ── Types ────────────────────────────────────────────────────

interface SymbioRiverMetric extends ChartMetric {
  symbio_candidate_id?: string | null;
  symbio_candidate_name?: string | null;
  symbio_candidate_activation?: string | null;
  symbio_candidate_parent_id?: string | null;
  symbio_candidate_parent_name?: string | null;
  symbio_generation?: number | null;
  symbio_activation_graph?: string | null;
  symbio_mutation_applied?: string | null;
  fitness_score?: number | null;
  architecture_diversity?: number | null;
  population_entropy?: number | null;
}

interface CandidateSegment {
  id: string;
  name: string;
  activation: string;
  generation: number;
  parentId: string | null;
  mutation: string | null;
  startStep: number;
  endStep: number;
  losses: number[];
  valLosses: number[];
  fitnesses: number[];
  gradNorms: number[];
  tokPerSec: number[];
  steps: number[];
  activationGraph: string | null;
  zEmbed: number;
  color: THREE.Color;
}

interface SwitchEvent {
  step: number;
  fromId: string | null;
  toId: string;
  toActivation: string;
  mutation: string | null;
  lossAtSwitch: number;
  fitnessDelta: number;
  generation: number;
}

interface FrontierPoint {
  step: number;
  bestLoss: number;
}

interface HoverInfo {
  x: number;
  y: number;
  segment: CandidateSegment;
  step: number;
  loss: number;
}

// ── Activation family colors ────────────────────────────────

const FAMILY_COLORS: Record<string, string> = {
  silu: "#60a5fa",
  gelu: "#a78bfa",
  relu: "#f59e0b",
  identity: "#94a3b8",
  square: "#fb923c",
  composed: "#e879f9",
};

function activationFamilyColor(activation: string): THREE.Color {
  const lower = activation.toLowerCase();
  for (const [family, hex] of Object.entries(FAMILY_COLORS)) {
    if (lower.includes(family) || lower.includes(family.slice(0, 2))) {
      return new THREE.Color(hex);
    }
  }
  // Detect identity as "id"
  if (lower.includes("id")) return new THREE.Color(FAMILY_COLORS.identity);
  if (lower.includes("sq")) return new THREE.Color(FAMILY_COLORS.square);
  return new THREE.Color("#e879f9");
}

function mutationColor(mutation: string | null): string {
  if (!mutation) return "#6b7280";
  if (mutation.includes("clone")) return "#34d399";
  if (mutation.includes("gate")) return "#f472b6";
  if (mutation.includes("prune")) return "#f87171";
  if (mutation.includes("swap")) return "#fbbf24";
  if (mutation.includes("residual")) return "#22d3ee";
  if (mutation.includes("add_term")) return "#a78bfa";
  if (mutation.includes("perturb")) return "#fb923c";
  return "#818cf8";
}

// ── Activation family Z embedding ────────────────────────────

function computeZEmbed(activation: string, graphStr: string | null): number {
  let silu = 0, gelu = 0, relu = 0, sq = 0, id = 0, depth = 0;

  if (graphStr) {
    try {
      const graph = JSON.parse(graphStr);
      const walk = (node: any, d: number) => {
        depth = Math.max(depth, d);
        if (node.type === "basis") {
          switch (node.op) {
            case "silu": silu += 1; break;
            case "gelu": gelu += 1; break;
            case "relu": relu += 1; break;
            case "identity": id += 1; break;
            case "square": sq += 1; break;
          }
        }
        if (node.child) walk(node.child, d + 1);
        if (node.left) walk(node.left, d + 1);
        if (node.right) walk(node.right, d + 1);
      };
      walk(graph, 0);
    } catch { /* use name-based fallback */ }
  }

  // Fallback from name
  if (silu + gelu + relu + sq + id === 0) {
    const lower = activation.toLowerCase();
    if (lower.includes("silu")) silu = 1;
    if (lower.includes("gelu")) gelu = 1;
    if (lower.includes("relu")) relu = 1;
    if (lower.includes("sq") || lower.includes("square")) sq = 1;
    if (lower.includes("id") || lower.includes("identity")) id = 1;
  }

  return 2 * silu + 1 * gelu - 1 * relu + 0.5 * sq + 0.2 * id + 0.1 * depth;
}

// ── Data extraction ────────────────────────────────────────

function extractSegments(metrics: SymbioRiverMetric[]): {
  segments: CandidateSegment[];
  switches: SwitchEvent[];
  frontier: FrontierPoint[];
} {
  const segments: CandidateSegment[] = [];
  const switches: SwitchEvent[] = [];
  const frontier: FrontierPoint[] = [];

  let currentId: string | null = null;
  let current: CandidateSegment | null = null;
  let bestLoss = Infinity;
  let prevFitness = 0;

  for (const m of metrics) {
    const id = m.symbio_candidate_id ?? null;
    if (!id) continue;

    // Track frontier
    if (m.loss < bestLoss) bestLoss = m.loss;
    frontier.push({ step: m.step, bestLoss });

    // Detect switch
    if (id !== currentId) {
      if (current) {
        current.endStep = metrics[metrics.indexOf(m) - 1]?.step ?? current.startStep;
        segments.push(current);
      }
      const activation = m.symbio_candidate_activation ?? "?";
      const graphStr = m.symbio_activation_graph ?? null;
      current = {
        id,
        name: m.symbio_candidate_name ?? id,
        activation,
        generation: m.symbio_generation ?? 0,
        parentId: m.symbio_candidate_parent_id ?? null,
        mutation: m.symbio_mutation_applied ?? null,
        startStep: m.step,
        endStep: m.step,
        losses: [],
        valLosses: [],
        fitnesses: [],
        gradNorms: [],
        tokPerSec: [],
        steps: [],
        activationGraph: graphStr,
        zEmbed: computeZEmbed(activation, graphStr),
        color: activationFamilyColor(activation),
      };

      const curFitness = m.fitness_score ?? 0;
      switches.push({
        step: m.step,
        fromId: currentId,
        toId: id,
        toActivation: activation,
        mutation: m.symbio_mutation_applied ?? null,
        lossAtSwitch: m.loss,
        fitnessDelta: curFitness - prevFitness,
        generation: m.symbio_generation ?? 0,
      });
      prevFitness = curFitness;
      currentId = id;
    }

    if (current) {
      current.losses.push(m.loss);
      current.steps.push(m.step);
      current.gradNorms.push(m.grad_norm);
      current.tokPerSec.push(m.tokens_per_sec);
      if (m.val_loss != null) current.valLosses.push(m.val_loss);
      if (m.fitness_score != null) current.fitnesses.push(m.fitness_score);
    }
  }

  if (current) {
    current.endStep = metrics[metrics.length - 1]?.step ?? current.startStep;
    segments.push(current);
  }

  return { segments, switches, frontier };
}

// ── Scale helpers ────────────────────────────────────────

function createScales(segments: CandidateSegment[], frontier: FrontierPoint[]) {
  let minStep = Infinity, maxStep = 0, minLoss = Infinity, maxLoss = 0;
  let minZ = Infinity, maxZ = -Infinity;

  for (const s of segments) {
    minStep = Math.min(minStep, s.startStep);
    maxStep = Math.max(maxStep, s.endStep);
    for (const l of s.losses) { minLoss = Math.min(minLoss, l); maxLoss = Math.max(maxLoss, l); }
    minZ = Math.min(minZ, s.zEmbed);
    maxZ = Math.max(maxZ, s.zEmbed);
  }
  for (const f of frontier) { minLoss = Math.min(minLoss, f.bestLoss); }

  const stepRange = maxStep - minStep || 1;
  const lossRange = maxLoss - minLoss || 1;
  const zRange = maxZ - minZ || 1;

  const sceneWidth = 20;
  const sceneHeight = 8;
  const sceneDepth = 6;

  return {
    x: (step: number) => ((step - minStep) / stepRange) * sceneWidth - sceneWidth / 2,
    y: (loss: number) => -((loss - minLoss) / lossRange) * sceneHeight + sceneHeight / 2,
    z: (zEmbed: number) => ((zEmbed - minZ) / zRange) * sceneDepth - sceneDepth / 2,
    minStep, maxStep, minLoss, maxLoss, sceneWidth, sceneHeight,
  };
}

// ── 3D Components ────────────────────────────────────────

function CandidateTube({
  segment, scales, time, onHover, onUnhover, dimmed,
}: {
  segment: CandidateSegment;
  scales: ReturnType<typeof createScales>;
  time: number;
  onHover: (seg: CandidateSegment, point: THREE.Vector3) => void;
  onUnhover: () => void;
  dimmed: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const points = useMemo(() => {
    if (segment.steps.length < 2) return null;
    const pts: THREE.Vector3[] = [];
    const z = scales.z(segment.zEmbed);
    for (let i = 0; i < segment.steps.length; i++) {
      pts.push(new THREE.Vector3(
        scales.x(segment.steps[i]),
        scales.y(segment.losses[i]),
        z,
      ));
    }
    return pts;
  }, [segment, scales]);

  const geometry = useMemo(() => {
    if (!points || points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points);
    // Radius based on average throughput
    const avgTps = segment.tokPerSec.reduce((s, v) => s + v, 0) / segment.tokPerSec.length;
    const radius = 0.02 + Math.min(0.08, avgTps / 30000);
    return new THREE.TubeGeometry(curve, Math.max(8, segment.steps.length), radius, 6, false);
  }, [points, segment.tokPerSec]);

  const material = useMemo(() => {
    const avgFitness = segment.fitnesses.length > 0
      ? segment.fitnesses.reduce((s, v) => s + v, 0) / segment.fitnesses.length : 0;
    const emissiveIntensity = 0.3 + Math.min(2, avgFitness * 20);
    return new THREE.MeshStandardMaterial({
      color: segment.color,
      emissive: segment.color,
      emissiveIntensity: dimmed ? 0.05 : emissiveIntensity,
      roughness: 0.4,
      metalness: 0.3,
      transparent: true,
      opacity: dimmed ? 0.15 : 0.85,
    });
  }, [segment.color, segment.fitnesses, dimmed]);

  if (!geometry) return null;

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      material={material}
      onPointerEnter={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        onHover(segment, e.point);
      }}
      onPointerLeave={onUnhover}
    />
  );
}

function FrontierSpine({
  frontier, scales, time,
}: {
  frontier: FrontierPoint[];
  scales: ReturnType<typeof createScales>;
  time: number;
}) {
  const points = useMemo(() => {
    if (frontier.length < 2) return [];
    // Subsample to max 200 points
    const stride = Math.max(1, Math.ceil(frontier.length / 200));
    return frontier
      .filter((_, i) => i % stride === 0)
      .map(f => new THREE.Vector3(
        scales.x(f.step),
        scales.y(f.bestLoss) + 0.3,
        0,
      ));
  }, [frontier, scales]);

  if (points.length < 2) return null;

  return (
    <Line
      points={points}
      color="#fbbf24"
      lineWidth={3}
      transparent
      opacity={0.9}
    />
  );
}

function SwitchPortal({
  event, scales,
}: {
  event: SwitchEvent;
  scales: ReturnType<typeof createScales>;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const x = scales.x(event.step);
  const y = scales.y(event.lossAtSwitch);
  const thickness = 0.02 + Math.min(0.1, Math.abs(event.fitnessDelta) * 5);
  const color = mutationColor(event.mutation);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += delta * 0.5;
    }
  });

  return (
    <mesh ref={meshRef} position={[x, y, 0]}>
      <torusGeometry args={[0.15, thickness, 8, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.8}
        transparent
        opacity={0.6}
      />
    </mesh>
  );
}

function DiversityFog({ metrics, scales }: { metrics: SymbioRiverMetric[]; scales: ReturnType<typeof createScales> }) {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    const stride = Math.max(1, Math.ceil(metrics.length / 300));

    for (let i = 0; i < metrics.length; i += stride) {
      const m = metrics[i];
      const entropy = m.population_entropy ?? 0;
      const diversity = m.architecture_diversity ?? 0;
      const density = entropy * diversity;
      if (density < 0.5) continue;

      const x = scales.x(m.step);
      const y = scales.y(m.loss);
      // Scatter points around the main data
      const count = Math.floor(density * 3);
      for (let j = 0; j < count; j++) {
        positions.push(
          x + (Math.random() - 0.5) * 1.5,
          y + (Math.random() - 0.5) * 1.5,
          (Math.random() - 0.5) * 4,
        );
        colors.push(0.4, 0.5, 0.9, 0.15);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return geo;
  }, [metrics, scales]);

  return (
    <points geometry={geometry}>
      <pointsMaterial
        color="#818cf8"
        size={0.04}
        transparent
        opacity={0.12}
        sizeAttenuation
      />
    </points>
  );
}

function LineageArcs({ segments, scales }: { segments: CandidateSegment[]; scales: ReturnType<typeof createScales> }) {
  const arcs = useMemo(() => {
    const segMap = new Map(segments.map(s => [s.id, s]));
    const out: { from: THREE.Vector3; to: THREE.Vector3; color: string }[] = [];
    for (const seg of segments) {
      if (!seg.parentId) continue;
      const parent = segMap.get(seg.parentId);
      if (!parent || parent.steps.length === 0) continue;
      const fromX = scales.x(parent.endStep);
      const fromY = scales.y(parent.losses[parent.losses.length - 1]);
      const fromZ = scales.z(parent.zEmbed);
      const toX = scales.x(seg.startStep);
      const toY = scales.y(seg.losses[0]);
      const toZ = scales.z(seg.zEmbed);
      out.push({
        from: new THREE.Vector3(fromX, fromY, fromZ),
        to: new THREE.Vector3(toX, toY, toZ),
        color: mutationColor(seg.mutation),
      });
    }
    return out;
  }, [segments, scales]);

  return (
    <>
      {arcs.map((arc, i) => {
        const mid = new THREE.Vector3().lerpVectors(arc.from, arc.to, 0.5);
        mid.y += 0.3;
        return (
          <Line
            key={i}
            points={[arc.from, mid, arc.to]}
            color={arc.color}
            lineWidth={1}
            transparent
            opacity={0.4}
            dashed
            dashSize={0.1}
            gapSize={0.05}
          />
        );
      })}
    </>
  );
}

function AxisLabels({ scales }: { scales: ReturnType<typeof createScales> }) {
  const stepLabels = useMemo(() => {
    const labels: { pos: THREE.Vector3; text: string }[] = [];
    const range = scales.maxStep - scales.minStep;
    const interval = Math.pow(10, Math.floor(Math.log10(range / 5)));
    const step = Math.max(interval, Math.ceil(range / 8 / interval) * interval);
    for (let s = Math.ceil(scales.minStep / step) * step; s <= scales.maxStep; s += step) {
      labels.push({
        pos: new THREE.Vector3(scales.x(s), -scales.sceneHeight / 2 - 0.5, 0),
        text: fmtNum(s),
      });
    }
    return labels;
  }, [scales]);

  return (
    <>
      {stepLabels.map((l, i) => (
        <Text
          key={i}
          position={l.pos}
          fontSize={0.2}
          color="#6b7280"
          anchorX="center"
          anchorY="top"
        >
          {l.text}
        </Text>
      ))}
      <Text
        position={[0, -scales.sceneHeight / 2 - 1, 0]}
        fontSize={0.18}
        color="#9ca3af"
        anchorX="center"
      >
        Training Step
      </Text>
    </>
  );
}

function AnimatedPulse({ segments, scales }: { segments: CandidateSegment[]; scales: ReturnType<typeof createScales> }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = segments.length;

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const t = clock.getElapsedTime();
    const dummy = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      const seg = segments[i];
      if (seg.steps.length < 2) continue;
      const progress = ((t * 0.3 + i * 0.1) % 1);
      const stepIdx = Math.floor(progress * (seg.steps.length - 1));
      const x = scales.x(seg.steps[stepIdx]);
      const y = scales.y(seg.losses[stepIdx]);
      const z = scales.z(seg.zEmbed);
      dummy.position.set(x, y, z);
      dummy.scale.setScalar(0.05 + 0.03 * Math.sin(t * 3 + i));
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={2} transparent opacity={0.6} />
    </instancedMesh>
  );
}

// ── Scene ────────────────────────────────────────────────

function RiverScene({
  segments, switches, frontier, metrics, scales,
  onHover, onUnhover, focusedId,
}: {
  segments: CandidateSegment[];
  switches: SwitchEvent[];
  frontier: FrontierPoint[];
  metrics: SymbioRiverMetric[];
  scales: ReturnType<typeof createScales>;
  onHover: (seg: CandidateSegment, point: THREE.Vector3) => void;
  onUnhover: () => void;
  focusedId: string | null;
}) {
  const timeRef = useRef(0);
  useFrame((_, delta) => { timeRef.current += delta; });

  return (
    <>
      <ambientLight intensity={0.3} />
      <directionalLight position={[5, 5, 5]} intensity={0.6} />
      <pointLight position={[0, 3, 0]} intensity={0.4} color="#fbbf24" />

      {/* Candidate tubes */}
      {segments.map(seg => (
        <CandidateTube
          key={seg.id + seg.startStep}
          segment={seg}
          scales={scales}
          time={timeRef.current}
          onHover={onHover}
          onUnhover={onUnhover}
          dimmed={focusedId !== null && focusedId !== seg.id}
        />
      ))}

      {/* Frontier spine */}
      <FrontierSpine frontier={frontier} scales={scales} time={timeRef.current} />

      {/* Switch portals */}
      {switches.slice(0, 50).map((sw, i) => (
        <SwitchPortal key={i} event={sw} scales={scales} />
      ))}

      {/* Lineage arcs */}
      <LineageArcs segments={segments} scales={scales} />

      {/* Diversity fog */}
      <DiversityFog metrics={metrics} scales={scales} />

      {/* Animated pulses */}
      <AnimatedPulse segments={segments} scales={scales} />

      {/* Axis labels */}
      <AxisLabels scales={scales} />

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.1}
        minDistance={3}
        maxDistance={30}
      />
    </>
  );
}

// ── Legend ────────────────────────────────────────────────

function RiverLegend({ segments }: { segments: CandidateSegment[] }) {
  const families = useMemo(() => {
    const map = new Map<string, { color: string; count: number; bestLoss: number }>();
    for (const s of segments) {
      const family = s.activation.split(/[×+·(]/)[0].trim();
      const existing = map.get(family);
      const best = Math.min(...s.losses);
      if (existing) {
        existing.count++;
        existing.bestLoss = Math.min(existing.bestLoss, best);
      } else {
        map.set(family, { color: "#" + s.color.getHexString(), count: 1, bestLoss: best });
      }
    }
    return [...map.entries()].sort((a, b) => a[1].bestLoss - b[1].bestLoss);
  }, [segments]);

  return (
    <div className="flex flex-wrap gap-2 text-[0.6rem]">
      {families.slice(0, 8).map(([name, info]) => (
        <div key={name} className="flex items-center gap-1.5 rounded border border-border/40 bg-surface-2/50 px-2 py-1">
          <div className="h-2 w-2 rounded-full" style={{ backgroundColor: info.color }} />
          <span className="text-text-secondary">{name}</span>
          <span className="text-text-muted">({info.count})</span>
        </div>
      ))}
      <div className="flex items-center gap-1.5 rounded border border-yellow-500/30 bg-yellow-500/5 px-2 py-1">
        <div className="h-0.5 w-3 rounded bg-yellow-400" />
        <span className="text-yellow-400">Frontier</span>
      </div>
      <div className="flex items-center gap-1.5 rounded border border-border/40 bg-surface-2/50 px-2 py-1">
        <div className="h-2 w-2 rounded-full border border-pink-400/60" />
        <span className="text-text-muted">Switch Portal</span>
      </div>
    </div>
  );
}

// ── Tooltip ────────────────────────────────────────────────

function RiverTooltip({ info }: { info: HoverInfo }) {
  const avgFitness = info.segment.fitnesses.length > 0
    ? info.segment.fitnesses.reduce((s, v) => s + v, 0) / info.segment.fitnesses.length : null;
  const avgTps = info.segment.tokPerSec.reduce((s, v) => s + v, 0) / info.segment.tokPerSec.length;

  return (
    <div
      className="pointer-events-none fixed z-50 rounded-lg border border-border-2 bg-surface-2 px-3 py-2 shadow-xl text-[0.62rem]"
      style={{ left: info.x + 16, top: info.y - 20 }}
    >
      <div className="mb-1 font-semibold text-white">{info.segment.name}</div>
      <div className="space-y-0.5 text-text-secondary">
        <div>Activation: <span className="text-white">{info.segment.activation}</span></div>
        <div>Generation: <span className="text-white">{info.segment.generation}</span></div>
        <div>Steps: <span className="text-white">{fmtNum(info.segment.startStep)}–{fmtNum(info.segment.endStep)}</span></div>
        <div>Best loss: <span className="text-yellow-400">{Math.min(...info.segment.losses).toFixed(4)}</span></div>
        {avgFitness != null && <div>Avg fitness: <span className="text-green-400">{avgFitness.toFixed(5)}</span></div>}
        <div>Avg tok/s: <span className="text-cyan-400">{Math.round(avgTps).toLocaleString()}</span></div>
        {info.segment.mutation && <div>Mutation: <span className="text-pink-400">{info.segment.mutation}</span></div>}
      </div>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────

export function SymbioRiver({ metrics }: { metrics: SymbioRiverMetric[] }) {
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const { segments, switches, frontier } = useMemo(() => extractSegments(metrics), [metrics]);
  const scales = useMemo(() => createScales(segments, frontier), [segments, frontier]);

  const symbioMetrics = useMemo(() => {
    return metrics.filter(m => m.symbio_candidate_id) as SymbioRiverMetric[];
  }, [metrics]);

  const handleHover = useCallback((seg: CandidateSegment, point: THREE.Vector3) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    // Approximate screen position from 3D point
    setHoverInfo({
      x: rect.left + rect.width / 2 + point.x * 25,
      y: rect.top + rect.height / 2 - point.y * 25,
      segment: seg,
      step: seg.startStep,
      loss: seg.losses[0],
    });
  }, []);

  const handleUnhover = useCallback(() => setHoverInfo(null), []);

  if (segments.length < 2) return null;

  return (
    <ChartPanel
      title="Symbio Evolution River"
      helpText="3D visualization of the symbiogenesis activation search. Each tube is a candidate lineage — X axis is training step, Y axis is loss, Z axis embeds activation families in space. Tube color = activation family, thickness = throughput, glow = fitness. The gold frontier spine tracks best-ever loss. Spinning rings mark candidate switches, colored by mutation type. Background fog density shows population diversity."
    >
      <div ref={containerRef} className="relative">
        <div className="h-[480px] w-full rounded-lg overflow-hidden bg-[#0a0a12]">
          <Canvas
            camera={{ position: [0, 2, 12], fov: 50 }}
            gl={{ antialias: true, alpha: false }}
            onPointerMissed={() => setFocusedId(null)}
          >
            <color attach="background" args={["#0a0a12"]} />
            <fog attach="fog" args={["#0a0a12", 15, 35]} />
            <RiverScene
              segments={segments}
              switches={switches}
              frontier={frontier}
              metrics={symbioMetrics}
              scales={scales}
              onHover={handleHover}
              onUnhover={handleUnhover}
              focusedId={focusedId}
            />
          </Canvas>
        </div>
        {hoverInfo && <RiverTooltip info={hoverInfo} />}
        <div className="mt-3">
          <RiverLegend segments={segments} />
        </div>
        <div className="mt-2 flex gap-4 text-[0.58rem] text-text-muted">
          <span>{segments.length} candidates across {switches.length} switches</span>
          <span>Generations: {Math.max(...segments.map(s => s.generation))}</span>
          <span>Best loss: {frontier.length > 0 ? frontier[frontier.length - 1].bestLoss.toFixed(4) : "-"}</span>
        </div>
      </div>
    </ChartPanel>
  );
}
