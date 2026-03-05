import { ArchBadge, Shape, ArchCard, ArchSectionHeading, PhaseBadge, KernelChip, InfraChip, Arrow, TransferStrip } from "@alpha/ui";

/* ── page ─────────────────────────────────────────────────────── */

export default function ArchitecturePage() {
  return (
    <div className="mx-auto max-w-[900px]">
      {/* Header */}
      <h1 className="mb-1 text-lg font-bold text-text-primary">Architecture</h1>
      <p className="mb-2 text-sm leading-relaxed text-text-secondary">
        CPU vs GPU computation map. Every component &mdash; tensors, autograd, model, tokenizers,
        training loop &mdash; is hand-written TypeScript with zero ML dependencies.
      </p>

      {/* Legend */}
      <div className="mb-8 flex flex-wrap gap-3">
        <ArchBadge variant="cpu">CPU &mdash; TypeScript</ArchBadge>
        <ArchBadge variant="gpu">GPU &mdash; Vulkan / SPIR-V</ArchBadge>
        <ArchBadge variant="dispatch">Backend Dispatch</ArchBadge>
        <span className="inline-flex items-center gap-1.5 rounded border border-dashed border-border-2 px-1.5 py-0.5 text-[0.62rem] text-text-muted">
          <span className="inline-block h-px w-4 border-t border-dashed border-text-muted" />
          Data Transfer
        </span>
      </div>

      {/* ═══════════════════════════════════════════ */}
      {/* PHASE 1: CPU-ONLY                          */}
      {/* ═══════════════════════════════════════════ */}

      <ArchSectionHeading color="#60a5fa">CPU-Only Operations</ArchSectionHeading>
      <p className="mb-4 text-[0.8rem] text-text-secondary">
        Always run on CPU regardless of tensor size. Orchestration, I/O, and control flow.
      </p>

      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
        <ArchCard variant="cpu" title="Data Loading">
          Tokenization (BPE / char / word), batch sampling, text I/O.{" "}
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">DataLoader.nextBatch()</code>{" "}
          picks random offsets &rarr; [B, T] token windows.
        </ArchCard>
        <ArchCard variant="cpu" title="RNG & Seeding">
          Deterministic seeded RNG (seed=42). Weight init N(0, 0.02), dropout masks, data sampling order.
        </ArchCard>
        <ArchCard variant="cpu" title="LR Schedule">
          Warmup: linear ramp over ~10% of steps.
          Cosine decay: <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">lr &times; 0.5(1+cos(&pi;&middot;decay))</code>.
        </ArchCard>
        <ArchCard variant="cpu" title="Checkpoint I/O">
          Binary v2: <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">ALPH</code> magic + JSON header + packed Float32.
          Saves params, optimizer m/v buffers, RNG state.
        </ArchCard>
        <ArchCard variant="cpu" title="Metrics & Logging">
          JSONL step metrics (loss, lr, grad norm, tokens/sec). Remote sync to{" "}
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">alpha.omegaai.dev</code>.
        </ArchCard>
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
          <span className="flex items-center gap-1.5 rounded border border-blue/20 bg-blue-bg/50 px-2 py-1 text-[0.65rem] text-blue">&larr; CPU path <ArchBadge variant="cpu">TypeScript</ArchBadge></span>
          <span className="flex items-center gap-1.5 rounded border border-yellow/20 bg-yellow-bg/50 px-2 py-1 text-[0.65rem] text-yellow">GPU path &rarr; <ArchBadge variant="gpu">SPIR-V</ArchBadge></span>
        </div>
      </div>

      <TransferStrip direction="upload">uploadBuffer &mdash; staging &rarr; device-local</TransferStrip>

      {/* ═══════════════════════════════════════════ */}
      {/* PHASE 2: FORWARD PASS                      */}
      {/* ═══════════════════════════════════════════ */}

      <ArchSectionHeading color="#4ade80">
        <PhaseBadge variant="forward">&triangleright; Forward Pass</PhaseBadge>
      </ArchSectionHeading>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* ─── CPU LANE ─── */}
        <div>
          <div className="mb-3 border-b-2 border-blue pb-2 text-[0.8rem] font-bold uppercase tracking-wider text-blue">
            CPU &mdash; TypeScript
          </div>
          <div className="space-y-2">
            <ArchCard variant="cpu" title="Token Embedding" shape="[B,T] → [B,T,nEmbd]">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">embedding(wte, tokens)</code> &mdash;
              gather rows from weight table. Tape records backward for gradient scatter.
            </ArchCard>
            <ArchCard variant="cpu" title="Position Embedding" shape="[B,T] → [B,T,nEmbd]">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">embedding(wpe, 0..T-1)</code> &mdash;
              positional encoding lookup.
            </ArchCard>
            <ArchCard variant="cpu" title="Add Embeddings" shape="[B,T,nEmbd]">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">tokEmb + posEmb</code> &mdash;
              element-wise sum. Small models stay on CPU.
            </ArchCard>
            <ArchCard variant="cpu" title="Causal Mask" shape="[T,T]">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">causalMask(T)</code> &mdash;
              lower triangle = 0, upper = &minus;&infin;. Always CPU (small, created once).
            </ArchCard>
            <ArchCard variant="cpu" title="Reshape / Transpose">
              View operations: reshape to multi-head [B, nHead, T, headDim], transpose for attention,
              concat heads. No computation &mdash; pointer reinterpretation.
            </ArchCard>
            <ArchCard variant="cpu" title="Residual Connections">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x = x + attnOut</code>,{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x = x + mlpOut</code> &mdash;
              identity shortcuts stabilizing deep nets.
            </ArchCard>
            <ArchCard variant="cpu" title="Tape Recording">
              Every op appends to global Tape: output Variable, input Variables, backward closure.
              Enables reverse-mode autodiff.
            </ArchCard>
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
              <ArchCard variant="gpu" title="Q, K, V Projections" shape="[B&middot;T,nEmbd]&sup2;">
                3&times; tiled matmul:{" "}
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x @ Wq</code>,{" "}
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x @ Wk</code>,{" "}
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">x @ Wv</code>.
                16&times;16 shared memory tiles.
              </ArchCard>
              <ArchCard variant="gpu" title="Attention Scores" shape="[B,nHead,T,T]">
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">Q @ K&#7488; / &radic;headDim</code> &mdash;
                batched matmul + fused scale. Largest tensor in the forward pass.
              </ArchCard>
              <ArchCard variant="gpu" title="Softmax" shape="[B&middot;nHead,T,T]">
                Fused row-wise kernel: max &rarr; subtract &rarr; exp &rarr; sum &rarr; normalize.
                One workgroup per row, tree reduction in shared memory.
              </ArchCard>
              <ArchCard variant="gpu" title="Value Weighting" shape="[B,nHead,T,headDim]">
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">scores @ V</code> &mdash;
                tiled matmul combining attention weights with value vectors.
              </ArchCard>
              <ArchCard variant="gpu" title="Output Projection" shape="[B&middot;T,nEmbd]">
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">concat(heads) @ Wo</code> &mdash; final attention matmul.
              </ArchCard>
            </div>

            {/* MLP */}
            <div className="mb-2 mt-4 text-[0.6rem] font-semibold uppercase tracking-widest text-text-muted">&mdash; Feed-Forward MLP &mdash;</div>
            <div className="space-y-2">
              <ArchCard variant="gpu" title="LayerNorm" shape="&times;2 per layer">
                Fused kernel: mean &rarr; variance &rarr; normalize &rarr; &gamma;x+&beta;. One workgroup per token. Pre-norm architecture.
              </ArchCard>
              <ArchCard variant="gpu" title="MLP FC1" shape="[B&middot;T,nEmbd] → [B&middot;T,4&middot;nEmbd]">
                Tiled matmul expanding to 4&times; hidden dimension.
              </ArchCard>
              <ArchCard variant="gpu" title="GELU Activation" shape="[B&middot;T,4&middot;nEmbd]">
                Fused tanh approximation:{" "}
                <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">0.5x(1+tanh(&radic;(2/&pi;)(x+0.044715x&sup3;)))</code>.
                Uses vec4 kernel when aligned.
              </ArchCard>
              <ArchCard variant="gpu" title="MLP FC2" shape="[B&middot;T,4&middot;nEmbd] → [B&middot;T,nEmbd]">
                Tiled matmul projecting back to model dimension.
              </ArchCard>
            </div>
          </div>

          <div className="mt-2 space-y-2">
            <ArchCard variant="gpu" title="Final LayerNorm" shape="[B,T,nEmbd]">
              Last normalization before vocabulary projection.
            </ArchCard>
            <ArchCard variant="gpu" title="LM Head Projection" shape="[B,T,vocabSize]">
              Matmul projecting to vocabulary logits.
            </ArchCard>
            <ArchCard variant="gpu" title="Cross-Entropy Loss" shape="→ scalar">
              Log-softmax + NLL:{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">-log(softmax(logits)[target])</code>{" "}
              averaged over B&times;T tokens.
            </ArchCard>
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

      <ArchSectionHeading color="#f87171">
        <PhaseBadge variant="backward">&triangleleft; Backward Pass</PhaseBadge>
      </ArchSectionHeading>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* CPU */}
        <div>
          <div className="mb-3 border-b-2 border-blue pb-2 text-[0.8rem] font-bold uppercase tracking-wider text-blue">
            CPU &mdash; Tape Walker
          </div>
          <div className="space-y-2">
            <ArchCard variant="cpu" title="Initialize Loss Gradient">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">loss.grad = ones(shape)</code> &mdash;
              seed the backward pass with dL/dL = 1.
            </ArchCard>
            <ArchCard variant="cpu" title="Reverse Tape Walk">
              Walk recorded tape from newest &rarr; oldest entry. Each entry calls its backward closure:{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">entry.backward(grad, backend)</code>.
              Gradients accumulate with += for multi-use variables (residuals).
            </ArchCard>
            <ArchCard variant="cpu" title="Broadcasting Undo">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">reduceBroadcast(grad, originalShape)</code> &mdash;
              sums over broadcast dims. E.g., [1, 512] broadcast to [64, 512] &rarr; sum over batch.
            </ArchCard>
            <ArchCard variant="cpu" title="Gradient Norm & Clipping">
              Collect all param gradients &rarr; compute global L2 norm.
              If <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">norm &gt; gradClip</code>: scale by{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">gradClip / norm</code>.
            </ArchCard>
          </div>
        </div>

        {/* GPU */}
        <div>
          <div className="mb-3 border-b-2 border-yellow pb-2 text-[0.8rem] font-bold uppercase tracking-wider text-yellow">
            GPU &mdash; Gradient Kernels
          </div>
          <div className="space-y-2">
            <ArchCard variant="gpu" title="CrossEntropy Backward">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">softmax(logits) &minus; oneHot(targets) / N</code>.
              Reuses forward softmax values.
            </ArchCard>
            <ArchCard variant="gpu" title="MatMul Backward" shape="&times;8 per layer">
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">dA = G @ B&#7488;</code>,{" "}
              <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">dB = A&#7488; @ G</code>.
              Each forward matmul produces 2 backward matmuls. Dominates backward compute.
            </ArchCard>
            <ArchCard variant="gpu" title="LayerNorm Backward">
              Full analytical gradient: normalized input, scale/bias updates. Fused kernel matching forward structure.
            </ArchCard>
            <ArchCard variant="gpu" title="Activation Backward">
              GELU: derivative of tanh approximation.
              Softmax: <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">s &times; (g &minus; &Sigma;(g&middot;s))</code>.
            </ArchCard>
            <ArchCard variant="gpu" title="Element-wise Backward">
              add / sub / mul / div / exp / log / sqrt &mdash; each has trivial derivative.
              Uses same vec4 kernel variants as forward.
            </ArchCard>
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════ */}
      {/* PHASE 4: OPTIMIZER                         */}
      {/* ═══════════════════════════════════════════ */}

      <ArchSectionHeading color="#f59e0b">
        <PhaseBadge variant="optimize">&orarr; Optimizer &mdash; AdamW</PhaseBadge>
      </ArchSectionHeading>
      <p className="mb-4 text-[0.8rem] text-text-secondary">
        All optimizer operations run on CPU. In-place parameter updates with decoupled weight decay.
      </p>

      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
        <ArchCard variant="cpu" title="Momentum">
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">m = &beta;&#8321;&middot;m + (1&minus;&beta;&#8321;)&middot;g</code><br />
          Exponential moving average of gradients. Per-parameter buffer.
        </ArchCard>
        <ArchCard variant="cpu" title="Variance">
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">v = &beta;&#8322;&middot;v + (1&minus;&beta;&#8322;)&middot;g&sup2;</code><br />
          EMA of squared gradients (RMSprop component).
        </ArchCard>
        <ArchCard variant="cpu" title="Bias Correction">
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">m&#770; = m/(1&minus;&beta;&#8321;&#7511;)</code><br />
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">v&#770; = v/(1&minus;&beta;&#8322;&#7511;)</code><br />
          Corrects for zero-init bias.
        </ArchCard>
        <ArchCard variant="cpu" title="Param Update">
          <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[0.65rem] text-text-primary">p &minus;= lr&middot;(m&#770;/(&radic;v&#770;+&epsilon;) + wd&middot;p)</code><br />
          Decoupled weight decay applied directly to params.
        </ArchCard>
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

      <ArchSectionHeading color="#ff9100">GPU Kernel Inventory &mdash; SPIR-V Compute Shaders</ArchSectionHeading>
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

      <ArchSectionHeading color="#ff9100">GPU Infrastructure &mdash; Vulkan Native Layer</ArchSectionHeading>
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
