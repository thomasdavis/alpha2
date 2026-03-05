"use client";

import { 
  Button, 
  Card, 
  CardHeader, 
  CardTitle, 
  CardContent, 
  CardFooter,
  Badge,
  Stat,
  DetailRow,
  Sparkline,
  ChartPanel,
  BaseMiniChart,
  Progress,
  InteractiveLossChart,
  SymbioSection,
  LayersSection,
  MethodBadge, Endpoint, Required, SectionHeading, Pre,
  ArchBadge, PhaseBadge, KernelChip
} from "@alpha/ui";
import { 
  Activity, 
  Cpu, 
  Zap, 
  Shield, 
  AlertTriangle, 
  CheckCircle2,
  Download,
  Terminal,
  Play,
  Settings,
  BarChart3,
  LineChart,
  Trophy,
  Book,
  Server,
  ActivitySquare,
  Microscope,
  Layers
} from "lucide-react";

export default function StyleGuidePage() {
  const mockSparkData = [1.2, 1.1, 1.3, 0.9, 0.8, 0.7, 0.75, 0.6, 0.55, 0.5, 0.48, 0.45];
  const mockMiniData = Array.from({ length: 50 }, (_, i) => ({
    step: i * 100,
    value: Math.sin(i / 5) * 10 + 50 + Math.random() * 5
  }));

  const mockChartMetrics = Array.from({ length: 100 }, (_, i) => {
    const layers: Record<string, number> = {
      "embed": 0.1 * Math.random(),
      "head": 0.05 * Math.random()
    };
    for(let l=0; l<6; l++) layers[String(l)] = Math.random() * Math.exp(-l/2);

    return {
      step: i * 50,
      loss: 2.5 * Math.exp(-i / 40) + Math.random() * 0.1,
      val_loss: i % 10 === 0 ? 2.6 * Math.exp(-i / 40) + Math.random() * 0.05 : null,
      lr: 0.0006 * (1 - i / 100),
      grad_norm: 0.1 + Math.random() * 0.2,
      tokens_per_sec: 12000 + Math.random() * 1000,
      ms_per_iter: 45,
      elapsed_ms: 4500,
      gpu_util_pct: 85,
      gpu_vram_used_mb: 4096,
      gpu_vram_total_mb: 16384,
      timing_fwd_ms: 15,
      timing_bwd_ms: 25,
      timing_optim_ms: 2,
      timing_data_ms: 1,
      timing_flush_ms: 2,
      timing_grad_norm_ms: 0.5,
      timing_grad_clip_ms: 0.5,
      gpu_ops_count: 120,
      // Symbio specific
      weight_entropy: 4.5 + Math.sin(i/10)*0.5,
      effective_rank: 128 - (i/2),
      free_energy: 2.1 - (i/50),
      population_entropy: 1.2,
      fitness_score: 0.8 + (i/200),
      complexity_score: 0.4,
      cusum_grad: Math.max(0, Math.sin(i/5)*4),
      cusum_clip: Math.max(0, Math.cos(i/8)*3),
      cusum_tps: 0.1,
      cusum_val: 0.2,
      // Layer specific
      per_layer_grad_norms: JSON.stringify(layers)
    };
  });

  const mockCheckpoints = [
    { step: 1000 },
    { step: 2500 },
    { step: 4000 }
  ];

  return (
    <div className="container mx-auto max-w-5xl px-4 py-12">
      <header className="mb-12 border-b border-border pb-8">
        <h1 className="text-4xl font-bold tracking-tight text-text-primary mb-2">Alpha Style Guide</h1>
        <p className="text-text-secondary text-lg">
          Core UI components for the Alpha training system. All components are built with React 19, Tailwind CSS v4, and Lucide Icons.
        </p>
      </header>

      {/* ── Buttons ─────────────────────────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-text-primary mb-6 border-l-4 border-accent pl-4 flex items-center gap-2">
          <Play className="h-5 w-5 text-accent" />
          Buttons
        </h2>
        <div className="grid gap-8">
          <Card>
            <CardHeader>
              <CardTitle>Variants</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-4">
              <Button variant="primary">Primary Button</Button>
              <Button variant="secondary">Secondary Button</Button>
              <Button variant="outline">Outline Button</Button>
              <Button variant="ghost">Ghost Button</Button>
              <Button variant="danger">Danger Button</Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Sizes</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap items-end gap-4">
              <Button size="sm">Small</Button>
              <Button size="md">Medium</Button>
              <Button size="lg">Large</Button>
              <Button size="icon" variant="outline"><Settings className="h-4 w-4" /></Button>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* ── Charts & Data ────────────────────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-text-primary mb-6 border-l-4 border-yellow pl-4 flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-yellow" />
          Charts & Data
        </h2>
        <div className="grid gap-6">
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Stat label="Loss" value="0.4521" color="text-yellow" tip="Current cross-entropy loss" />
            <Stat label="Throughput" value="12.4k" sub="tokens/sec" color="text-green" tip="Tokens processed per second" />
            <Stat label="VRAM" value="14.2 GB" sub="85% utilized" color="text-blue" />
            <Stat label="MFU" value="42.1%" color="text-rose-400" tip="Model FLOPS Utilization" />
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Sparklines */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  Sparklines
                  <Activity className="h-4 w-4 text-text-muted" />
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-text-secondary">Active Training</span>
                  <Sparkline data={mockSparkData} variant="success" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-text-secondary">Validation Loss</span>
                  <Sparkline data={[2, 1.8, 1.9, 1.5, 1.6, 1.4, 1.3]} variant="blue" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-text-secondary">Gradient Spikes</span>
                  <Sparkline data={[0.1, 0.1, 0.8, 0.1, 0.2, 0.9, 0.1]} variant="danger" />
                </div>
              </CardContent>
            </Card>

            {/* Detail Rows */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  Detail Table
                  <Terminal className="h-4 w-4 text-text-muted" />
                </CardTitle>
              </CardHeader>
              <CardContent>
                <DetailRow label="Optimizer" value="AdamW" tip="Weight Decay Adaptive Moments" />
                <DetailRow label="Learning Rate" value="6.0e-4" />
                <DetailRow label="Weight Decay" value="0.1" />
                <DetailRow label="Batch Size" value="512" />
                <DetailRow label="Block Size" value="1024" />
              </CardContent>
            </Card>
          </div>

          {/* Progress Section */}
          <Card>
            <CardHeader>
              <CardTitle>Training Progress</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between text-[0.65rem] uppercase font-bold text-text-muted">
                  <span>Iteration 45,000 / 50,000</span>
                  <span>90%</span>
                </div>
                <Progress value={90} variant="default" className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-[0.65rem] uppercase font-bold text-text-muted">
                  <span>Checkpoint Sync</span>
                  <span className="text-green">Healthy</span>
                </div>
                <Progress value={100} variant="success" className="h-1.5" />
              </div>
            </CardContent>
          </Card>

          {/* New Charts Subsection */}
          <div className="grid gap-6">
            <ChartPanel title="Optimization Telemetry" helpText="Visualization of internal training dynamics using the BaseMiniChart component.">
              <BaseMiniChart 
                data={mockMiniData} 
                title="Gradient Norm (L2)" 
                height={180} 
                formatValue={(v) => v.toFixed(1)} 
              />
            </ChartPanel>
          </div>

          {/* Complex Charts Subsection */}
          <div className="mt-8 space-y-6">
            <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest text-text-primary">
              <ActivitySquare className="h-4 w-4 text-accent" />
              Complex Interactive Charts
            </div>
            
            <Card>
              <CardHeader>
                <CardTitle>Interactive Loss Chart</CardTitle>
              </CardHeader>
              <CardContent>
                <InteractiveLossChart 
                  metrics={mockChartMetrics as any} 
                  checkpoints={mockCheckpoints} 
                />
              </CardContent>
              <CardFooter className="bg-surface-2/30 text-[0.65rem] text-text-muted italic">
                Note: This is a high-performance canvas implementation with interactive markers and tooltips.
              </CardFooter>
            </Card>
          </div>
        </div>
      </section>

      {/* ── Symbiogenesis ────────────────────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-text-primary mb-6 border-l-4 border-purple-400 pl-4 flex items-center gap-2">
          <Microscope className="h-5 w-5 text-purple-400" />
          Evolutionary Analytics
        </h2>
        <SymbioSection metrics={mockChartMetrics as any} />
      </section>

      {/* ── Per-Layer Analysis ─────────────────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-text-primary mb-6 border-l-4 border-cyan-400 pl-4 flex items-center gap-2">
          <Layers className="h-5 w-5 text-cyan-400" />
          Transformer Layer Analysis
        </h2>
        <LayersSection metrics={mockChartMetrics as any} />
      </section>

      {/* ── Badges ─────────────────────────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-text-primary mb-6 border-l-4 border-green pl-4 flex items-center gap-2">
          <Trophy className="h-5 w-5 text-green" />
          Badges & Status
        </h2>
        <Card>
          <CardHeader>
            <CardTitle>Status & Categorization</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-4">
            <Badge variant="default">Default</Badge>
            <Badge variant="secondary">Secondary</Badge>
            <Badge variant="outline">Outline</Badge>
            <Badge variant="success" className="gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Completed
            </Badge>
            <Badge variant="warning" className="gap-1">
              <AlertTriangle className="h-3 w-3" />
              Stale
            </Badge>
            <Badge variant="danger" className="gap-1">
              <Shield className="h-3 w-3" />
              Failed
            </Badge>
            <Badge variant="blue" className="gap-1">
              <Activity className="h-3 w-3" />
              Active
            </Badge>
          </CardContent>
        </Card>
      </section>

      {/* ── Domain Specific Components ───────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-text-primary mb-6 border-l-4 border-blue pl-4 flex items-center gap-2">
          <Book className="h-5 w-5 text-blue" />
          Documentation & Architecture
        </h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>API Documentation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-2">
                <MethodBadge method="GET" />
                <MethodBadge method="POST" />
                <Endpoint path="/v1/models" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-text-primary">Some Field</span>
                <Required />
              </div>
              <Pre>{"{\n  \"model\": \"alpha-1\"\n}"}</Pre>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Architecture Diagram Elements</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-2">
                <ArchBadge variant="cpu">CPU</ArchBadge>
                <ArchBadge variant="gpu">GPU</ArchBadge>
                <ArchBadge variant="dispatch">Dispatch</ArchBadge>
              </div>
              <div className="flex items-center gap-2">
                <PhaseBadge variant="forward">Forward</PhaseBadge>
                <PhaseBadge variant="backward">Backward</PhaseBadge>
              </div>
              <KernelChip name="gemm_coop">
                Cooperative matrix multiplication kernel using tensor cores.
              </KernelChip>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* ── Typography & Colors ────────────────────────────────────────── */}
      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-text-primary mb-6 border-l-4 border-purple-400 pl-4 flex items-center gap-2">
          <Zap className="h-5 w-5 text-purple-400" />
          Typography & Colors
        </h2>
        <Card>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-12 pt-6">
            <div className="space-y-4">
              <h3 className="text-text-muted uppercase text-[0.65rem] font-bold tracking-widest">Theme Palette</h3>
              <div className="grid grid-cols-4 gap-4">
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-accent border border-black/5" />
                  <span className="text-[0.6rem] text-text-muted font-mono">accent</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-green border border-black/5" />
                  <span className="text-[0.6rem] text-text-muted font-mono">green</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-yellow border border-black/5" />
                  <span className="text-[0.6rem] text-text-muted font-mono">yellow</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-red border border-black/5" />
                  <span className="text-[0.6rem] text-text-muted font-mono">red</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-bg border border-border" />
                  <span className="text-[0.6rem] text-text-muted font-mono">bg</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-surface border border-border" />
                  <span className="text-[0.6rem] text-text-muted font-mono">surface</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-surface-2 border border-border" />
                  <span className="text-[0.6rem] text-text-muted font-mono">surface-2</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-border border border-border-2" />
                  <span className="text-[0.6rem] text-text-muted font-mono">border</span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-text-muted uppercase text-[0.65rem] font-bold tracking-widest">Text Hierarchy</h3>
              <div className="space-y-4">
                <div>
                  <div className="text-text-primary text-3xl font-bold">Heading 1</div>
                  <span className="text-[0.6rem] text-text-muted font-mono">3xl / bold / text-primary</span>
                </div>
                <div>
                  <div className="text-text-primary text-xl font-semibold">Heading 2</div>
                  <span className="text-[0.6rem] text-text-muted font-mono">xl / semibold / text-primary</span>
                </div>
                <div>
                  <div className="text-text-secondary text-base">Standard paragraph text.</div>
                  <span className="text-[0.6rem] text-text-muted font-mono">base / text-secondary</span>
                </div>
                <div>
                  <div className="text-text-muted text-sm font-mono uppercase tracking-widest">Label Text</div>
                  <span className="text-[0.6rem] text-text-muted font-mono">sm / mono / uppercase / text-muted</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
