/* ── helper components ────────────────────────────────────────── */

function Badge({ variant, children }: { variant: "cpu" | "gpu" | "dispatch"; children: React.ReactNode }) {
  const styles = {
    cpu: "bg-blue-bg text-blue",
    gpu: "bg-yellow-bg text-yellow",
    dispatch: "bg-[#2a1a3a] text-purple-400",
  };
  return (
    <span className={`inline-block rounded px-1.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wide ${styles[variant]}`}>
      {children}
    </span>
  );
}

function Shape({ children }: { children: React.ReactNode }) {
  return <span className="ml-1 text-[0.65rem] text-text-muted font-mono">{children}</span>;
}

function Card({
  variant,
  title,
  shape,
  children,
  className,
}: {
  variant: "cpu" | "gpu" | "dispatch";
  title: string;
  shape?: string;
  children: React.ReactNode;
  className?: string;
}) {
  const border = { cpu: "border-l-blue", gpu: "border-l-yellow", dispatch: "border-l-purple-400" };
  return (
    <div className={`rounded-md border border-border bg-surface pl-0 ${className ?? ""}`}>
      <div className={`border-l-2 ${border[variant]} rounded-md px-3 py-2.5`}>
        <div className="mb-1 flex items-center gap-2 text-[0.8rem] font-semibold text-white">
          {title}
          {shape && <Shape>{shape}</Shape>}
        </div>
        <div className="text-[0.75rem] leading-relaxed text-text-secondary">{children}</div>
      </div>
    </div>
  );
}

function SectionHeading({ color, children }: { color?: string; children: React.ReactNode }) {
  return (
    <h2
      className="mb-3 mt-10 border-b border-border pb-2 text-[1.1rem] font-semibold text-white"
      style={color ? { color } : undefined}
    >
      {children}
    </h2>
  );
}

function PhaseBadge({ variant, children }: { variant: "forward" | "backward" | "optimize"; children: React.ReactNode }) {
  const styles = {
    forward: "bg-green-bg text-green",
    backward: "bg-red-bg text-red",
    optimize: "bg-yellow-bg text-yellow",
  };
  return (
    <span className={`inline-block rounded px-2 py-1 text-[0.7rem] font-bold uppercase tracking-wider ${styles[variant]}`}>
      {children}
    </span>
  );
}

function KernelChip({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2.5 transition-colors hover:border-border-2">
      <div className="mb-0.5 text-[0.75rem] font-semibold text-yellow">{name}</div>
      <div className="text-[0.65rem] leading-relaxed text-text-secondary">{children}</div>
    </div>
  );
}

function InfraChip({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2.5">
      <div className="mb-0.5 text-[0.75rem] font-semibold text-yellow">{name}</div>
      <div className="text-[0.65rem] leading-relaxed text-text-secondary">{children}</div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex justify-center py-1">
      <svg width="16" height="20" viewBox="0 0 16 20" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-text-muted">
        <line x1="8" y1="0" x2="8" y2="16" />
        <polyline points="4,12 8,17 12,12" />
      </svg>
    </div>
  );
}

function TransferStrip({ direction, children }: { direction: "upload" | "download"; children: React.ReactNode }) {
  const bg = direction === "upload" ? "from-blue-bg/50 to-yellow-bg/50" : "from-yellow-bg/50 to-blue-bg/50";
  return (
    <div className={`my-4 flex items-center justify-center gap-3 rounded-md border border-dashed border-border-2 bg-gradient-to-r ${bg} px-4 py-2.5 text-[0.7rem] text-text-secondary`}>
      {direction === "upload" ? (
        <>
          <Badge variant="cpu">CPU</Badge>
          <span className="flex gap-0.5">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.15s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.3s" }} />
          </span>
          <span>{children}</span>
          <span className="flex gap-0.5">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.45s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.6s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.75s" }} />
          </span>
          <Badge variant="gpu">GPU</Badge>
        </>
      ) : (
        <>
          <Badge variant="gpu">GPU</Badge>
          <span className="flex gap-0.5">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.15s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.3s" }} />
          </span>
          <span>{children}</span>
          <span className="flex gap-0.5">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.45s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.6s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.75s" }} />
          </span>
          <Badge variant="cpu">CPU</Badge>
        </>
      )}
    </div>
  );
}

/* ── page ─────────────────────────────────────────────────────── */

export default function ArchitecturePage() {
  return (
    <div className="mx-auto max-w-[900px]">
      {/* Header */}
      <h1 className="mb-1 text-lg font-bold text-white">Architecture</h1>
      <p className="mb-2 text-sm leading-relaxed text-text-secondary">
        CPU vs GPU computation map. Every component &mdash; tensors, autograd, model, tokenizers,
        training loop &mdash; is hand-written TypeScript with zero ML dependencies.
      </p>

      {/* Legend */}
      <div className="mb-8 flex flex-wrap gap-3">
        <Badge variant="cpu">CPU &mdash; TypeScript</Badge>
        <Badge variant="gpu">GPU &mdash; Vulkan / SPIR-V</Badge>
        <Badge variant="dispatch">Backend Dispatch</Badge>
        <span className="inline-flex items-center gap-1.5 rounded border border-dashed border-border-2 px-1.5 py-0.5 text-[0.62rem] text-text-muted">
          <span className="inline-block h-px w-4 border-t border-dashed border-text-muted" />
          Data Transfer
        </span>
      </div>

      {/* ═══════════════════════════════════════════ */}
      {/* PHASE 1: CPU-ONLY                          */}
      {/* ═══════════════════════════════════════════ */}

      <SectionHeading color="#60a5fa">CPU-Only Operations</SectionHeading>
      <p className="mb-4 text-[0.8rem] text-text-secondary">
        Always run on CPU regardless of tensor size. Orchestration, I/O, and control flow.
      </p>

      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
        <Card variant="cpu" title="Data Loading">
          Tokenization (BPE / char / word), batch sampling, text I/O.{" "}
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">DataLoader.nextBatch()</code>{" "}
          picks random offsets &rarr; [B, T] token windows.
        </Card>
        <Card variant="cpu" title="RNG & Seeding">
          Deterministic seeded RNG (seed=42). Weight init N(0, 0.02), dropout masks, data sampling order.
        </Card>
        <Card variant="cpu" title="LR Schedule">
          Warmup: linear ramp over ~10% of steps.
          Cosine decay: <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">lr &times; 0.5(1+cos(&pi;&middot;decay))</code>.
        </Card>
        <Card variant="cpu" title="Checkpoint I/O">
          Binary v2: <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">ALPH</code> magic + JSON header + packed Float32.
          Saves params, optimizer m/v buffers, RNG state.
        </Card>
        <Card variant="cpu" title="Metrics & Logging">
          JSONL step metrics (loss, lr, grad norm, tokens/sec). Remote sync to{" "}
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">alpha.omegaai.dev</code>.
        </Card>
      </div>

      {/* ═══════════════════════════════════════════ */}
      {/* DISPATCH DECISION                          */}
      {/* ═══════════════════════════════════════════ */}

      <Arrow />

      <div className="my-4 flex flex-col items-center">
        <div className="flex h-24 w-24 rotate-45 items-center justify-center rounded-md border-2 border-purple-400 bg-[#1a0a2a] shadow-[0_0_20px_rgba(168,85,247,0.15)]">
          <div className="-rotate-45 text-center">
            <div className="text-[0.65rem] font-bold uppercase tracking-wider text-purple-400">Backend</div>
            <div className="text-[0.6rem] font-bold uppercase tracking-wider text-purple-400">Dispatch</div>
          </div>
        </div>
        <p className="mt-3 text-center text-[0.75rem] text-text-secondary">
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-purple-400">&ge; 1,000,000 elements</code> &rarr; GPU &nbsp;&nbsp;|&nbsp;&nbsp;
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-purple-400">&lt; 1M</code> &rarr; CPU
        </p>
        <div className="mt-2 flex gap-4">
          <span className="flex items-center gap-1.5 rounded border border-blue/20 bg-blue-bg/50 px-2 py-1 text-[0.65rem] text-blue">&larr; CPU path <Badge variant="cpu">TypeScript</Badge></span>
          <span className="flex items-center gap-1.5 rounded border border-yellow/20 bg-yellow-bg/50 px-2 py-1 text-[0.65rem] text-yellow">GPU path &rarr; <Badge variant="gpu">SPIR-V</Badge></span>
        </div>
      </div>

      <TransferStrip direction="upload">uploadBuffer &mdash; staging &rarr; device-local</TransferStrip>

      {/* ═══════════════════════════════════════════ */}
      {/* PHASE 2: FORWARD PASS                      */}
      {/* ═══════════════════════════════════════════ */}

      <SectionHeading color="#4ade80">
        <PhaseBadge variant="forward">&triangleright; Forward Pass</PhaseBadge>
      </SectionHeading>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* ─── CPU LANE ─── */}
        <div>
          <div className="mb-3 border-b-2 border-blue pb-2 text-[0.8rem] font-bold uppercase tracking-wider text-blue">
            CPU &mdash; TypeScript
          </div>
          <div className="space-y-2">
            <Card variant="cpu" title="Token Embedding" shape="[B,T] → [B,T,nEmbd]">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">embedding(wte, tokens)</code> &mdash;
              gather rows from weight table. Tape records backward for gradient scatter.
            </Card>
            <Card variant="cpu" title="Position Embedding" shape="[B,T] → [B,T,nEmbd]">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">embedding(wpe, 0..T-1)</code> &mdash;
              positional encoding lookup.
            </Card>
            <Card variant="cpu" title="Add Embeddings" shape="[B,T,nEmbd]">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">tokEmb + posEmb</code> &mdash;
              element-wise sum. Small models stay on CPU.
            </Card>
            <Card variant="cpu" title="Causal Mask" shape="[T,T]">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">causalMask(T)</code> &mdash;
              lower triangle = 0, upper = &minus;&infin;. Always CPU (small, created once).
            </Card>
            <Card variant="cpu" title="Reshape / Transpose">
              View operations: reshape to multi-head [B, nHead, T, headDim], transpose for attention,
              concat heads. No computation &mdash; pointer reinterpretation.
            </Card>
            <Card variant="cpu" title="Residual Connections">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x = x + attnOut</code>,{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x = x + mlpOut</code> &mdash;
              identity shortcuts stabilizing deep nets.
            </Card>
            <Card variant="cpu" title="Tape Recording">
              Every op appends to global Tape: output Variable, input Variables, backward closure.
              Enables reverse-mode autodiff.
            </Card>
          </div>
        </div>

        {/* ─── GPU LANE ─── */}
        <div>
          <div className="mb-3 border-b-2 border-yellow pb-2 text-[0.8rem] font-bold uppercase tracking-wider text-yellow">
            GPU &mdash; Vulkan Compute
          </div>

          {/* Transformer block */}
          <div className="rounded-lg border border-dashed border-purple-400/40 bg-[#0f0a18]/30 p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[0.7rem] font-bold uppercase tracking-wider text-purple-400">Transformer Block</span>
              <span className="rounded bg-[#2a1a3a] px-1.5 py-0.5 text-[0.6rem] font-bold text-purple-400">&times;N layers</span>
            </div>

            {/* Attention */}
            <div className="mb-2 text-[0.6rem] font-semibold uppercase tracking-widest text-text-muted">&mdash; Multi-Head Attention &mdash;</div>
            <div className="space-y-2">
              <Card variant="gpu" title="Q, K, V Projections" shape="[B&middot;T,nEmbd]&sup2;">
                3&times; tiled matmul:{" "}
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x @ Wq</code>,{" "}
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x @ Wk</code>,{" "}
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x @ Wv</code>.
                16&times;16 shared memory tiles.
              </Card>
              <Card variant="gpu" title="Attention Scores" shape="[B,nHead,T,T]">
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">Q @ K&#7488; / &radic;headDim</code> &mdash;
                batched matmul + fused scale. Largest tensor in the forward pass.
              </Card>
              <Card variant="gpu" title="Softmax" shape="[B&middot;nHead,T,T]">
                Fused row-wise kernel: max &rarr; subtract &rarr; exp &rarr; sum &rarr; normalize.
                One workgroup per row, tree reduction in shared memory.
              </Card>
              <Card variant="gpu" title="Value Weighting" shape="[B,nHead,T,headDim]">
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">scores @ V</code> &mdash;
                tiled matmul combining attention weights with value vectors.
              </Card>
              <Card variant="gpu" title="Output Projection" shape="[B&middot;T,nEmbd]">
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">concat(heads) @ Wo</code> &mdash; final attention matmul.
              </Card>
            </div>

            {/* MLP */}
            <div className="mb-2 mt-4 text-[0.6rem] font-semibold uppercase tracking-widest text-text-muted">&mdash; Feed-Forward MLP &mdash;</div>
            <div className="space-y-2">
              <Card variant="gpu" title="LayerNorm" shape="&times;2 per layer">
                Fused kernel: mean &rarr; variance &rarr; normalize &rarr; &gamma;x+&beta;. One workgroup per token. Pre-norm architecture.
              </Card>
              <Card variant="gpu" title="MLP FC1" shape="[B&middot;T,nEmbd] → [B&middot;T,4&middot;nEmbd]">
                Tiled matmul expanding to 4&times; hidden dimension.
              </Card>
              <Card variant="gpu" title="GELU Activation" shape="[B&middot;T,4&middot;nEmbd]">
                Fused tanh approximation:{" "}
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">0.5x(1+tanh(&radic;(2/&pi;)(x+0.044715x&sup3;)))</code>.
                Uses vec4 kernel when aligned.
              </Card>
              <Card variant="gpu" title="MLP FC2" shape="[B&middot;T,4&middot;nEmbd] → [B&middot;T,nEmbd]">
                Tiled matmul projecting back to model dimension.
              </Card>
            </div>
          </div>

          <div className="mt-2 space-y-2">
            <Card variant="gpu" title="Final LayerNorm" shape="[B,T,nEmbd]">
              Last normalization before vocabulary projection.
            </Card>
            <Card variant="gpu" title="LM Head Projection" shape="[B,T,vocabSize]">
              Matmul projecting to vocabulary logits.
            </Card>
            <Card variant="gpu" title="Cross-Entropy Loss" shape="→ scalar">
              Log-softmax + NLL:{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">-log(softmax(logits)[target])</code>{" "}
              averaged over B&times;T tokens.
            </Card>
          </div>
        </div>
      </div>

      {/* Transformer summary */}
      <div className="mt-4 rounded-md border border-border bg-surface px-4 py-3 text-[0.75rem] leading-relaxed text-text-secondary">
        <span className="font-semibold text-white">Per transformer layer:</span>{" "}
        8 matmuls + 2 layernorms + 1 softmax + 1 GELU + 2 residual adds + mask fill
      </div>

      <TransferStrip direction="download">readBuffer &mdash; loss scalar + activations for backward</TransferStrip>

      {/* ═══════════════════════════════════════════ */}
      {/* PHASE 3: BACKWARD PASS                     */}
      {/* ═══════════════════════════════════════════ */}

      <SectionHeading color="#f87171">
        <PhaseBadge variant="backward">&triangleleft; Backward Pass</PhaseBadge>
      </SectionHeading>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* CPU */}
        <div>
          <div className="mb-3 border-b-2 border-blue pb-2 text-[0.8rem] font-bold uppercase tracking-wider text-blue">
            CPU &mdash; Tape Walker
          </div>
          <div className="space-y-2">
            <Card variant="cpu" title="Initialize Loss Gradient">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">loss.grad = ones(shape)</code> &mdash;
              seed the backward pass with dL/dL = 1.
            </Card>
            <Card variant="cpu" title="Reverse Tape Walk">
              Walk recorded tape from newest &rarr; oldest entry. Each entry calls its backward closure:{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">entry.backward(grad, backend)</code>.
              Gradients accumulate with += for multi-use variables (residuals).
            </Card>
            <Card variant="cpu" title="Broadcasting Undo">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">reduceBroadcast(grad, originalShape)</code> &mdash;
              sums over broadcast dims. E.g., [1, 512] broadcast to [64, 512] &rarr; sum over batch.
            </Card>
            <Card variant="cpu" title="Gradient Norm & Clipping">
              Collect all param gradients &rarr; compute global L2 norm.
              If <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">norm &gt; gradClip</code>: scale by{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">gradClip / norm</code>.
            </Card>
          </div>
        </div>

        {/* GPU */}
        <div>
          <div className="mb-3 border-b-2 border-yellow pb-2 text-[0.8rem] font-bold uppercase tracking-wider text-yellow">
            GPU &mdash; Gradient Kernels
          </div>
          <div className="space-y-2">
            <Card variant="gpu" title="CrossEntropy Backward">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">softmax(logits) &minus; oneHot(targets) / N</code>.
              Reuses forward softmax values.
            </Card>
            <Card variant="gpu" title="MatMul Backward" shape="&times;8 per layer">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">dA = G @ B&#7488;</code>,{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">dB = A&#7488; @ G</code>.
              Each forward matmul produces 2 backward matmuls. Dominates backward compute.
            </Card>
            <Card variant="gpu" title="LayerNorm Backward">
              Full analytical gradient: normalized input, scale/bias updates. Fused kernel matching forward structure.
            </Card>
            <Card variant="gpu" title="Activation Backward">
              GELU: derivative of tanh approximation.
              Softmax: <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">s &times; (g &minus; &Sigma;(g&middot;s))</code>.
            </Card>
            <Card variant="gpu" title="Element-wise Backward">
              add / sub / mul / div / exp / log / sqrt &mdash; each has trivial derivative.
              Uses same vec4 kernel variants as forward.
            </Card>
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════ */}
      {/* PHASE 4: OPTIMIZER                         */}
      {/* ═══════════════════════════════════════════ */}

      <SectionHeading color="#f59e0b">
        <PhaseBadge variant="optimize">&orarr; Optimizer &mdash; AdamW</PhaseBadge>
      </SectionHeading>
      <p className="mb-4 text-[0.8rem] text-text-secondary">
        All optimizer operations run on CPU. In-place parameter updates with decoupled weight decay.
      </p>

      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
        <Card variant="cpu" title="Momentum">
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">m = &beta;&#8321;&middot;m + (1&minus;&beta;&#8321;)&middot;g</code><br />
          Exponential moving average of gradients. Per-parameter buffer.
        </Card>
        <Card variant="cpu" title="Variance">
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">v = &beta;&#8322;&middot;v + (1&minus;&beta;&#8322;)&middot;g&sup2;</code><br />
          EMA of squared gradients (RMSprop component).
        </Card>
        <Card variant="cpu" title="Bias Correction">
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">m&#770; = m/(1&minus;&beta;&#8321;&#7511;)</code><br />
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">v&#770; = v/(1&minus;&beta;&#8322;&#7511;)</code><br />
          Corrects for zero-init bias.
        </Card>
        <Card variant="cpu" title="Param Update">
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">p &minus;= lr&middot;(m&#770;/(&radic;v&#770;+&epsilon;) + wd&middot;p)</code><br />
          Decoupled weight decay applied directly to params.
        </Card>
      </div>

      <div className="my-4 flex items-center justify-center gap-2 text-[0.7rem] text-text-muted">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="12,2 12,14 4,14 4,2" />
          <polyline points="7,11 4,14 7,17" />
          <polyline points="1,5 4,2 7,5" />
        </svg>
        <span className="uppercase tracking-widest">Next training step</span>
      </div>

      {/* ═══════════════════════════════════════════ */}
      {/* GPU KERNEL INVENTORY                       */}
      {/* ═══════════════════════════════════════════ */}

      <SectionHeading color="#ff9100">GPU Kernel Inventory &mdash; SPIR-V Compute Shaders</SectionHeading>
      <p className="mb-4 text-[0.8rem] text-text-secondary">
        25+ hand-written compute shaders generated by the TypeScript SPIR-V assembler.
      </p>

      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
        <KernelChip name="Element-wise">
          <code className="text-[0.6rem] text-text-primary">add sub mul div neg exp log sqrt scale</code><br />
          + vec4 variants (4&times; throughput via 128-bit loads)
        </KernelChip>
        <KernelChip name="Activations">
          <code className="text-[0.6rem] text-text-primary">relu gelu silu</code><br />
          + vec4 variants. GELU uses fused tanh approx.
        </KernelChip>
        <KernelChip name="Reductions">
          <code className="text-[0.6rem] text-text-primary">sum_reduce max_reduce</code><br />
          Parallel tree reduction in shared memory. log&#8322;(WG_SIZE) barriers.
        </KernelChip>
        <KernelChip name="Fused Softmax">
          Row-wise: max &rarr; exp(x&minus;max) &rarr; sum &rarr; normalize.
          One workgroup per row. No intermediate buffers.
        </KernelChip>
        <KernelChip name="Fused LayerNorm">
          mean &rarr; var &rarr; (x&minus;&mu;)/&radic;(&sigma;&sup2;+&epsilon;) &rarr; &gamma;x+&beta;.
          One workgroup per token.
        </KernelChip>
        <KernelChip name="Tiled MatMul">
          16&times;16 shared memory tiles. Loop over K dimension.
          Barrier-synchronized cooperative loads.
        </KernelChip>
        <KernelChip name="Fused Mul-Add">
          <code className="text-[0.6rem] text-text-primary">D[i] = A[i]&times;B[i] + C[i]</code><br />
          Single-pass 3-input kernel.
        </KernelChip>
        <KernelChip name="F16 Storage">
          <code className="text-[0.6rem] text-text-primary">add_f16 sub_f16 mul_f16 div_f16 neg_f16 exp_f16</code><br />
          Compute in f32, store as f16. 2&times; memory savings.
        </KernelChip>
      </div>

      {/* ═══════════════════════════════════════════ */}
      {/* GPU INFRASTRUCTURE                         */}
      {/* ═══════════════════════════════════════════ */}

      <SectionHeading color="#ff9100">GPU Infrastructure &mdash; Vulkan Native Layer</SectionHeading>
      <p className="mb-4 text-[0.8rem] text-text-secondary">
        Zero external dependencies. Custom Vulkan bridge, SPIR-V assembler, and memory management.
      </p>

      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
        <InfraChip name="Vulkan Native Addon">
          C (~2000 LOC), N-API bridge. Dynamic Vulkan loading via{" "}
          <code className="text-[0.6rem] text-text-primary">dlopen</code> &mdash; no SDK headers needed.
          Physical device selection + compute queue.
        </InfraChip>
        <InfraChip name="SPIR-V Assembler">
          Hand-written in TypeScript (~2500 LOC). Binary assembler &mdash; no{" "}
          <code className="text-[0.6rem] text-text-primary">glslc</code> or{" "}
          <code className="text-[0.6rem] text-text-primary">glslangValidator</code>.
          Types, decorations, control flow, GLSL.std.450 extended instructions.
        </InfraChip>
        <InfraChip name="Slab Memory Allocator">
          Bump-pointer allocation in 64MB slabs (up to 512MB per slab).
          Buffer pooling + recycling via{" "}
          <code className="text-[0.6rem] text-text-primary">FinalizationRegistry</code>.
        </InfraChip>
        <InfraChip name="Timeline Semaphores">
          Vulkan 1.2 timeline semaphores for async dispatch.
          Record many ops &rarr; wait only when results needed. Pipelined execution.
        </InfraChip>
        <InfraChip name="Compute Graph Batching">
          Lazy evaluation &mdash; ops recorded to pending queue. Auto-flush at 64 ops.
          Single command buffer submit: N&times;100&mu;s &rarr; 1&times;100&mu;s + N&times;2&mu;s.
        </InfraChip>
        <InfraChip name="Auto-Tuned Workgroups">
          Benchmarks &#123;64, 128, 256, 512&#125; on init via 256K element add.
          Optimal WG_SIZE cached for session. Vec4 kernels when elements % 4 == 0.
        </InfraChip>
      </div>

      {/* Footer */}
      <div className="mt-10 border-t border-border pt-4 text-center text-[0.7rem] text-text-muted">
        <span className="font-semibold text-text-secondary">Alpha</span> &mdash;
        40 CPU ops &middot; 25+ GPU kernels &middot; 0 external ML dependencies &middot; Pure TypeScript + Hand-written SPIR-V
      </div>
    </div>
  );
}
