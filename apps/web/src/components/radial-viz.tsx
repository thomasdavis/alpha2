"use client";

import { useRef, useMemo, useCallback, useEffect, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";

// ── Types ──────────────────────────────────────────────────

interface MetricPoint {
  step: number;
  loss: number;
  val_loss: number | null;
  lr: number;
  grad_norm: number;
  tokens_per_sec: number;
  ms_per_iter: number;
  weight_entropy?: number | null;
  effective_rank?: number | null;
  free_energy?: number | null;
  fitness_score?: number | null;
  clip_coef?: number | null;
  symbio_candidate_activation?: string | null;
  symbio_generation?: number | null;
  architecture_diversity?: number | null;
  cusum_alerts?: number | null;
  mi_input_repr?: number | null;
  mi_repr_output?: number | null;
}

const ACTIVATION_COLORS: Record<string, THREE.Color> = {
  gelu: new THREE.Color("#60a5fa"),
  silu: new THREE.Color("#34d399"),
  relu: new THREE.Color("#f59e0b"),
  swiglu: new THREE.Color("#a78bfa"),
  universal: new THREE.Color("#f472b6"),
  kan_spline: new THREE.Color("#22d3ee"),
};
const DEFAULT_COLOR = new THREE.Color("#666666");

// ── Helpers ────────────────────────────────────────────────

function normalize(arr: number[]): number[] {
  if (arr.length === 0) return [];
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = max - min || 1;
  return arr.map((v) => (v - min) / range);
}

function ema(arr: number[], alpha = 0.1): number[] {
  const result: number[] = [];
  let s = arr[0] ?? 0;
  for (const v of arr) {
    s = alpha * v + (1 - alpha) * s;
    result.push(s);
  }
  return result;
}

// ── Core Ring Geometry ─────────────────────────────────────

function DataRing({
  data,
  radius,
  width,
  yValues,
  colors,
  opacity = 0.8,
  pulseSpeed = 0,
  label,
}: {
  data: MetricPoint[];
  radius: number;
  width: number;
  yValues: number[];
  colors: THREE.Color[];
  opacity?: number;
  pulseSpeed?: number;
  label: string;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const n = data.length;
  const segments = Math.max(n, 64);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions: number[] = [];
    const colorAttrs: number[] = [];
    const indices: number[] = [];

    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const angle = t * Math.PI * 2 - Math.PI / 2;
      const idx = Math.min(Math.floor(t * n), n - 1);
      const y = yValues[idx] ?? 0;
      const h = y * width;
      const c = colors[idx] ?? DEFAULT_COLOR;

      // Inner vertex
      const r0 = radius - width * 0.5;
      positions.push(Math.cos(angle) * r0, h * 0.3, Math.sin(angle) * r0);
      colorAttrs.push(c.r, c.g, c.b);

      // Outer vertex (elevated by value)
      const r1 = radius + width * 0.5;
      positions.push(Math.cos(angle) * r1, h, Math.sin(angle) * r1);
      colorAttrs.push(c.r * 1.3, c.g * 1.3, c.b * 1.3);

      if (i < segments) {
        const base = i * 2;
        indices.push(base, base + 1, base + 2);
        indices.push(base + 1, base + 3, base + 2);
      }
    }

    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colorAttrs, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }, [segments, n, yValues, colors, radius, width]);

  useFrame(({ clock }) => {
    if (meshRef.current && pulseSpeed > 0) {
      const s = 1 + Math.sin(clock.elapsedTime * pulseSpeed) * 0.02;
      meshRef.current.scale.set(s, s, s);
    }
  });

  return (
    <group>
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          transparent
          opacity={opacity}
          side={THREE.DoubleSide}
          roughness={0.4}
          metalness={0.3}
        />
      </mesh>
      <Text
        position={[0, -0.15, radius + width]}
        fontSize={0.12}
        color="#888"
        anchorX="center"
        anchorY="middle"
        rotation={[-Math.PI / 2, 0, 0]}
      >
        {label}
      </Text>
    </group>
  );
}

// ── Particle Field (gradient flow) ─────────────────────────

function GradientParticles({ data }: { data: MetricPoint[] }) {
  const particlesRef = useRef<THREE.Points>(null);
  const count = 2000;

  const { positions, velocities, particleColors } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = 0.5 + Math.random() * 4;
      pos[i * 3] = Math.cos(angle) * r;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 2;
      pos[i * 3 + 2] = Math.sin(angle) * r;

      vel[i * 3] = (Math.random() - 0.5) * 0.01;
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.005;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.01;

      // Color by position (inner=hot, outer=cool)
      const t = r / 4.5;
      col[i * 3] = 0.3 + (1 - t) * 0.7;
      col[i * 3 + 1] = 0.1 + t * 0.5;
      col[i * 3 + 2] = 0.5 + t * 0.5;
    }
    return { positions: pos, velocities: vel, particleColors: col };
  }, []);

  useFrame(({ clock }) => {
    if (!particlesRef.current) return;
    const posArr = particlesRef.current.geometry.attributes.position.array as Float32Array;
    const t = clock.elapsedTime;

    // Get current training intensity from last data point
    const last = data[data.length - 1];
    const intensity = last ? Math.min(last.grad_norm / 5, 1) : 0.3;

    for (let i = 0; i < count; i++) {
      const ix = i * 3, iy = i * 3 + 1, iz = i * 3 + 2;
      const x = posArr[ix], z = posArr[iz];
      const r = Math.sqrt(x * x + z * z);
      const angle = Math.atan2(z, x);

      // Orbital motion + radial oscillation driven by training data
      const orbitalSpeed = 0.3 + intensity * 0.5;
      const newAngle = angle + orbitalSpeed * 0.016;
      const radialOsc = Math.sin(t * 2 + r * 3) * 0.02 * intensity;

      posArr[ix] = Math.cos(newAngle) * (r + radialOsc);
      posArr[iy] += velocities[iy] + Math.sin(t * 3 + i * 0.1) * 0.002;
      posArr[iz] = Math.sin(newAngle) * (r + radialOsc);

      // Bound Y
      if (Math.abs(posArr[iy]) > 1.5) posArr[iy] *= 0.95;
      // Respawn if too far
      if (r > 5 || r < 0.3) {
        const newR = 0.5 + Math.random() * 4;
        const a = Math.random() * Math.PI * 2;
        posArr[ix] = Math.cos(a) * newR;
        posArr[iz] = Math.sin(a) * newR;
      }
    }
    particlesRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[particleColors, 3]} />
      </bufferGeometry>
      <pointsMaterial
        vertexColors
        size={0.03}
        transparent
        opacity={0.6}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// ── Central Orb (loss state) ───────────────────────────────

function CentralOrb({ data }: { data: MetricPoint[] }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const t = clock.elapsedTime;
    const last = data[data.length - 1];
    const lossNorm = last ? Math.max(0.3, 1 - (last.loss - 2) / 8) : 0.5;

    // Pulse with loss
    const s = 0.25 + lossNorm * 0.15 + Math.sin(t * 2) * 0.02;
    meshRef.current.scale.set(s, s, s);

    // Color shift: high loss = red, low loss = cyan
    const mat = meshRef.current.material as THREE.MeshStandardMaterial;
    const hue = lossNorm * 0.5; // 0=red, 0.5=cyan
    mat.color.setHSL(hue, 0.8, 0.6);
    mat.emissive.setHSL(hue, 1, 0.3);

    if (glowRef.current) {
      const gs = s * 2.5 + Math.sin(t * 1.5) * 0.1;
      glowRef.current.scale.set(gs, gs, gs);
      (glowRef.current.material as THREE.MeshBasicMaterial).opacity = 0.08 + Math.sin(t * 2) * 0.03;
    }
  });

  return (
    <group>
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial
          roughness={0.2}
          metalness={0.8}
          emissiveIntensity={0.5}
        />
      </mesh>
      <mesh ref={glowRef}>
        <sphereGeometry args={[1, 16, 16]} />
        <meshBasicMaterial
          color="#4488ff"
          transparent
          opacity={0.1}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

// ── Generation Pulse Rings ─────────────────────────────────

function GenerationRings({ data }: { data: MetricPoint[] }) {
  const genChanges = useMemo(() => {
    const changes: { step: number; gen: number }[] = [];
    let prevGen = -1;
    for (const m of data) {
      const g = m.symbio_generation ?? -1;
      if (g !== prevGen && g >= 0) {
        changes.push({ step: m.step, gen: g });
        prevGen = g;
      }
    }
    return changes;
  }, [data]);

  return (
    <group>
      {genChanges.map(({ gen }, i) => (
        <PulseRing key={i} radius={1.2 + gen * 0.35} gen={gen} delay={i * 0.5} />
      ))}
    </group>
  );
}

function PulseRing({ radius, gen, delay }: { radius: number; gen: number; delay: number }) {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (!ringRef.current) return;
    const t = (clock.elapsedTime - delay) % 8;
    const opacity = t > 0 && t < 3 ? Math.sin((t / 3) * Math.PI) * 0.3 : 0.05;
    (ringRef.current.material as THREE.MeshBasicMaterial).opacity = opacity;
  });

  const hue = (gen * 0.08) % 1;
  const color = new THREE.Color().setHSL(hue, 0.7, 0.5);

  return (
    <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[radius - 0.02, radius + 0.02, 64]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.1}
        side={THREE.DoubleSide}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </mesh>
  );
}

// ── Spike Markers (grad norm spikes) ───────────────────────

function SpikeMarkers({ data }: { data: MetricPoint[] }) {
  const spikes = useMemo(() => {
    const gradNorms = data.map((d) => d.grad_norm);
    const smoothed = ema(gradNorms, 0.05);
    const results: { angle: number; height: number; color: THREE.Color }[] = [];

    for (let i = 1; i < data.length; i++) {
      const ratio = gradNorms[i] / (smoothed[i] || 1);
      if (ratio > 3) {
        const angle = (i / data.length) * Math.PI * 2 - Math.PI / 2;
        const act = data[i].symbio_candidate_activation;
        results.push({
          angle,
          height: Math.min(ratio * 0.3, 2),
          color: (act ? ACTIVATION_COLORS[act] : undefined) ?? DEFAULT_COLOR,
        });
      }
    }
    return results;
  }, [data]);

  return (
    <group>
      {spikes.map((s, i) => (
        <mesh key={i} position={[Math.cos(s.angle) * 3.5, s.height * 0.5, Math.sin(s.angle) * 3.5]}>
          <cylinderGeometry args={[0.01, 0.04, s.height, 6]} />
          <meshStandardMaterial
            color={s.color}
            emissive={s.color}
            emissiveIntensity={0.8}
            transparent
            opacity={0.7}
          />
        </mesh>
      ))}
    </group>
  );
}

// ── Scene ──────────────────────────────────────────────────

function Scene({ data }: { data: MetricPoint[] }) {
  const n = data.length;
  if (n < 2) return null;

  // Precompute normalized series & colors
  const lossNorm = normalize(data.map((d) => d.loss));
  const lrNorm = normalize(data.map((d) => d.lr));
  const gradNorm = normalize(data.map((d) => Math.min(d.grad_norm, 10)));
  const tpsNorm = normalize(data.map((d) => d.tokens_per_sec));
  const entropyNorm = normalize(data.map((d) => d.weight_entropy ?? 0));
  const diversityNorm = normalize(data.map((d) => d.architecture_diversity ?? 0));
  const clipNorm = normalize(data.map((d) => Math.min(1, d.clip_coef ?? 1)));
  const miNorm = normalize(data.map((d) => d.mi_input_repr ?? 0));

  // Smoothed versions for outer rings
  const lossSmooth = ema(lossNorm, 0.05);
  const gradSmooth = ema(gradNorm, 0.05);

  // Activation-based colors for loss ring
  const activationColors = data.map((d) => {
    const act = d.symbio_candidate_activation;
    return (act ? ACTIVATION_COLORS[act] : undefined) ?? DEFAULT_COLOR;
  });

  // Gradient-based colors (red=high, blue=low)
  const gradColors = gradNorm.map((v) => {
    const c = new THREE.Color();
    c.setHSL(0.6 - v * 0.6, 0.9, 0.4 + v * 0.3);
    return c;
  });

  // LR colors (warm ramp)
  const lrColors = lrNorm.map((v) => {
    const c = new THREE.Color();
    c.setHSL(0.08 + v * 0.1, 0.8, 0.3 + v * 0.4);
    return c;
  });

  // Throughput colors (green spectrum)
  const tpsColors = tpsNorm.map((v) => {
    const c = new THREE.Color();
    c.setHSL(0.3, 0.6 + v * 0.3, 0.3 + v * 0.3);
    return c;
  });

  // Entropy colors (purple spectrum)
  const entropyColors = entropyNorm.map((v) => {
    const c = new THREE.Color();
    c.setHSL(0.75, 0.5 + v * 0.4, 0.3 + v * 0.3);
    return c;
  });

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[5, 5, 5]} intensity={0.8} color="#aaccff" />
      <pointLight position={[-5, -3, -5]} intensity={0.4} color="#ffaa66" />
      <pointLight position={[0, 3, 0]} intensity={0.6} color="#ffffff" />

      {/* Central orb — loss state indicator */}
      <CentralOrb data={data} />

      {/* Ring 1: Loss (innermost) — color-coded by activation */}
      <DataRing
        data={data} radius={1.2} width={0.3}
        yValues={lossSmooth} colors={activationColors}
        opacity={0.85} pulseSpeed={1.5} label="loss"
      />

      {/* Ring 2: Gradient norm */}
      <DataRing
        data={data} radius={1.8} width={0.25}
        yValues={gradSmooth} colors={gradColors}
        opacity={0.7} pulseSpeed={2} label="grad"
      />

      {/* Ring 3: Learning rate */}
      <DataRing
        data={data} radius={2.3} width={0.2}
        yValues={lrNorm} colors={lrColors}
        opacity={0.6} label="lr"
      />

      {/* Ring 4: Throughput */}
      <DataRing
        data={data} radius={2.75} width={0.2}
        yValues={tpsNorm} colors={tpsColors}
        opacity={0.55} label="tok/s"
      />

      {/* Ring 5: Weight entropy (outer) */}
      {data.some((d) => d.weight_entropy != null) && (
        <DataRing
          data={data} radius={3.2} width={0.18}
          yValues={entropyNorm} colors={entropyColors}
          opacity={0.5} label="entropy"
        />
      )}

      {/* Ring 6: Clip coefficient */}
      {data.some((d) => d.clip_coef != null) && (
        <DataRing
          data={data} radius={3.6} width={0.15}
          yValues={clipNorm} colors={gradColors}
          opacity={0.45} label="clip"
        />
      )}

      {/* Generation pulse rings */}
      <GenerationRings data={data} />

      {/* Gradient spike markers */}
      <SpikeMarkers data={data} />

      {/* Particle field */}
      <GradientParticles data={data} />

      {/* Ground reference plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]}>
        <ringGeometry args={[0.5, 4.5, 128]} />
        <meshBasicMaterial
          color="#111122"
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
        />
      </mesh>

      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        autoRotate
        autoRotateSpeed={0.5}
        minDistance={2}
        maxDistance={12}
        maxPolarAngle={Math.PI * 0.85}
      />
    </>
  );
}

// ── Export ──────────────────────────────────────────────────

export function RadialTrainingViz({ metrics }: { metrics: MetricPoint[] }) {
  const [isClient, setIsClient] = useState(false);
  useEffect(() => setIsClient(true), []);

  if (!isClient || metrics.length < 5) return null;

  return (
    <div className="relative h-[500px] w-full rounded-lg border border-border bg-[#060612] overflow-hidden">
      {/* Legend overlay */}
      <div className="absolute top-3 left-3 z-10 flex flex-wrap gap-2 pointer-events-none">
        {[
          { label: "Loss", color: "#60a5fa" },
          { label: "Grad Norm", color: "#f59e0b" },
          { label: "Learning Rate", color: "#fb923c" },
          { label: "Throughput", color: "#34d399" },
          { label: "Entropy", color: "#c084fc" },
        ].map(({ label, color }) => (
          <span key={label} className="flex items-center gap-1 rounded bg-black/50 px-1.5 py-0.5 text-[0.55rem] text-text-muted backdrop-blur-sm">
            <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
            {label}
          </span>
        ))}
      </div>

      {/* Activation legend */}
      <div className="absolute top-3 right-3 z-10 flex flex-wrap gap-1.5 pointer-events-none">
        {Object.entries({ gelu: "#60a5fa", silu: "#34d399", relu: "#f59e0b", swiglu: "#a78bfa", universal: "#f472b6", kan_spline: "#22d3ee" }).map(
          ([name, color]) => (
            <span key={name} className="rounded bg-black/50 px-1.5 py-0.5 text-[0.55rem] backdrop-blur-sm" style={{ color }}>
              {name}
            </span>
          )
        )}
      </div>

      {/* Step counter */}
      <div className="absolute bottom-3 left-3 z-10 pointer-events-none">
        <span className="rounded bg-black/60 px-2 py-1 font-mono text-xs text-text-muted backdrop-blur-sm">
          step {metrics[metrics.length - 1]?.step?.toLocaleString() ?? "—"} | {metrics.length} points
        </span>
      </div>

      {/* Interaction hint */}
      <div className="absolute bottom-3 right-3 z-10 pointer-events-none">
        <span className="rounded bg-black/40 px-2 py-0.5 text-[0.5rem] text-text-muted/50 backdrop-blur-sm">
          drag to rotate | scroll to zoom
        </span>
      </div>

      <Canvas
        camera={{ position: [3, 4, 5], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
      >
        <Scene data={metrics} />
      </Canvas>
    </div>
  );
}
