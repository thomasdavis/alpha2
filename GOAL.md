# GOAL — Bring Alpha back to life: a from-scratch chatty model, trained by Alpha's own code, published on Hugging Face

**Set:** 2026-07-22 · **Owner:** ajax + Codex (handoff from Claude) · **Status:** ACTIVE
**Budget:** $70.21 RunPod prepaid credit (hard ceiling; no per-token API spend anywhere in this program)

---

## 1. Mission

Train a small **chatty** language model **entirely with Alpha's own from-scratch stack** — the TypeScript
tensor library, the tape autograd, the hand-generated SPIR-V kernels, and the Helios Vulkan backend —
**GPU-resident at all times**, on RunPod, and publish it to Hugging Face as a **standard, zero-custom-code
model** (`LlamaForCausalLM` + safetensors + tokenizer.json) that anyone can load with `transformers`.

**The soul constraint (operator, 2026-07-22): Alpha is all about writing everything from scratch. Every
training FLOP goes through Alpha's own code on GPU.** No PyTorch training, no CUDA libraries, no external
training frameworks. External tools are permitted only OUTSIDE the training loop: dataset download/cleaning,
checkpoint conversion verification, and publishing.

Why this is worth doing: nobody else has a from-scratch TS/Vulkan stack that trains real transformers.
The deliverable is simultaneously (a) proof the framework works end-to-end at useful scale, and (b) a
genuinely usable artifact — a tiny conversational model with an honest model card, loadable by anyone.

## 2. What is already PROVEN (2026-07-22 beachhead — do not re-derive)

1. **Vulkan compute works on RunPod community pods.** Proven empirically on a $0.22/hr RTX 3090
   (`vkCreateInstance: 0`, discrete GPU enumerated, API 1.4.312). The recipe (now in
   `scripts/runpod_bootstrap.sh`):
   - Pods run with `NVIDIA_DRIVER_CAPABILITIES=compute,utility` and RunPod's CDI runtime **ignores** the
     env override → no graphics libs, no ICD. Fix: download the NVIDIA `.run` installer **matching the
     host driver exactly** (`nvidia-smi --query-gpu=driver_version`), stub `modprobe/rmmod/insmod/lsmod/depmod`
     with `exit 0` shims, run installer with `--silent --no-kernel-modules`, then write the **EGL headless
     ICD** (`{"ICD":{"library_path":"libEGL_nvidia.so.0", ...}}`) and `export VK_ICD_FILENAMES=` to it,
     `unset DISPLAY`. The stock `nvidia_icd.json` (libGLX_nvidia) does NOT work headless — GIPA returns NULL.
   - Community hosts have flaky egress (port-80 apt dead, github unreachable on some) — bootstrap must not
     depend on apt or github; rsync everything from the box; nodejs.org + download.nvidia.com (443) worked.
2. **The monorepo builds clean** on the OVH box (19/19 turbo tasks, Node 22.23, npm install clean).
3. **RunPod account live**: $70.21 credit, runpodctl 2.6.1 + GraphQL/REST verified, SSH key at
   `~/.runpod/ssh/runpodctl-ssh-key`, pod create→ssh→terminate loop proven. Live prices (community):
   A5000 24GB $0.16 · 3090 24GB $0.22 · A40 48GB $0.30-0.35 · 4090 24GB $0.34.
4. **HF publish access works**: `ajaxdavis` authed on the box, write probed (repo create+delete OK).
5. **Recon corpus**: 13 subsystem/research reports + critic (2026-07-22 workflow) — key numbers baked
   into this doc. Full copies in the session scratchpad; the numbers that matter are restated here.

## 3. Definition of DONE

**D1. `ajaxdavis/alpha-<N>m-base`** and **D2. `ajaxdavis/alpha-<N>m-chat`** on Hugging Face:
standard `config.json` (`architectures:["LlamaForCausalLM"]`), `model.safetensors`, `tokenizer.json` +
`tokenizer_config.json` + `chat_template.jinja`, honest model card (from-scratch provenance, data mix,
full eval table, limitations). Verification: `pipeline("text-generation", "ajaxdavis/alpha-<N>m-chat")`
produces formatted chat output on a clean machine with **zero custom code**.

**D3. The chat bar** (frozen eval, greedy decoding, run before upload):
- ≥95/100 fixed chat prompts: responds in assistant role, terminates with EOS, no user-role leakage.
- Repetition: 4-gram repeat rate < 0.20 on the sample suite; no degenerate loops in 100 samples.
- Coherence: grades like "talking to a small child" are ACCEPTABLE and reported honestly; gibberish is not.
  (Calibration: prior Alpha chat runs = word salad at val-loss ~4.7; karpathy nanochat 561M @ 11.2B tokens
  = "kindergartener"; SmolLM2-135M @ 2T = grammatical but limited. We aim for "clearly conversational".)
- 200 closed-book sanity questions: report the true score, whatever it is. No benchmark training data.

**D4. Training provenance**: every checkpoint traceable to a run config + git hash + data manifest;
the whole run reproducible from repo scripts. All flops through Helios (`--backend=helios`,
`HELIOS_NO_FALLBACK` smoke-checked at boot; CPU fallback allowed only for sub-`minGpuSize` scraps).

## 4. Stages and GATES (sequential; a stage's gate must pass before spending on the next)

### Stage 0 — Finish the beachhead ✅ COMPLETE 2026-07-22 (actual: $0.36)
- [x] Vulkan proof on RunPod (2026-07-22).
- [x] **Gate G0 PASSED (2026-07-22)**: Helios smoke train on community RTX 3090 — 60/60 steps, 1.33M
      params, loss 7.28→7.05, grad norms ~0.4, **zero non-finite events**, **~40-42K tok/s**, 415 gpu
      ops/step, log header `gpu: NVIDIA GeForce RTX 3090 (NVIDIA)` (not llvmpipe), **DGC enabled**
      (VK_EXT_device_generated_commands works on driver 580) + BDA + coop-matmul active, checkpoint
      saved + pulled to `/mnt/donto-data/alpha-runs/g0-smoke-20260722/`. Notable: 3090 gave ~40K tok/s
      at a shape that did 65K on L4 — untuned (default WG_SIZE 128, no per-GPU profile); Stage 2 tunes.
- [x] `scripts/runpod_bootstrap.sh` committed; pod create→bootstrap→train→pull→terminate documented in
      `docs/RUNPOD.md`. Full-repo rsync from box takes ~30-45 min under box I/O load — sync
      `packages/ apps/` first (small) if in a hurry; consider a pod-side tarball cache later.

### Stage 1 — Trustworthy engine (box-heavy, ≈$3 GPU for parity runs)
**ENGINE + NVIDIA PARITY COMPLETE 2026-07-22** (remaining: the G1 1K-step pilot): deps modernized to
latest (TS 7, vitest 4, Next 16,
effect 3.22; npm audit 5→1-low via overrides); gradcheck harness landed (9b63685) — 42 per-op central-
difference checks + whole-model gradchecks (swiglu/gelu/universal/kan_spline) + AdamW-vs-reference +
GPU-gated parity suite; **REAL BUG found+fixed: cpu_ref.sum(keepdims=true) corrupted broadcast backward
grads on non-last axes**; lmHead no-decay + tokenizer --vocabSize + fp16-auto-enable + train-nanochat lr
bugs all fixed; secrets scrubbed (Discord webhook REVOKED, .env untracked; ElevenLabs key still needs
dashboard rotation by user).
The recon found: documented **2-7% NaN-gradient steps** (Helios×SwiGLU, root cause never fixed), fp16
diverges immediately, `lmHead` weight-decay exclusion bug (`"lmHead.weight"` vs actual name `"lmHead"`),
wte/wpe missing from no-decay in older audits, spike-skip machinery papering over real numerical bugs,
and **zero Helios test files**.
- [x] CPU↔Helios parity harness: fixed weights + fixed batch →
      compare logits, loss, every gradient, one full AdamW step, 100 deterministic steps; fail on first
      non-finite. Cover: matmul (all variants in use), softmax, layernorm/rmsnorm, silu/siluMul, CE
      fwd/bwd, embedding bwd, flash-attention fwd/bwd vs standard path. **NVIDIA gate PASSED on RTX 3090
      pod `d5m7h1v0kr0zd4`: 44/44 tests, 0 failures**; evidence:
      `/mnt/donto-data/alpha-runs/gpu-gates-20260722/gpu-gates-pass-44-20260722.log`. The run exposed and
      fixed real f16-clone corruption, missing multi-output write barriers, non-atomic repeated-token
      embedding gradients, no-op vec4 in-place accumulation, masked negative zero, unsafe partial flash
      tiles, and padded-buffer readback length leakage. Box regression remains 178 pass / 44 GPU-skipped;
      `tsc -b` clean.
- [x] Root-cause the historical SwiGLU NaN rate with the harness and NVIDIA pilot. The parity run found
      concrete gradient corruption in multi-output dispatch barriers, repeated-token embedding scatter,
      vec4 in-place accumulation, and f16 cloning; all were fixed rather than masked. The G1 pilot then
      crossed the full warmup/peak-LR region and all 1,000 steps with no numerical skip.
- [ ] Fix known bugs: lmHead no-decay name; audit no-decay set (norms + embeddings excluded from wd);
      delete/ignore doc-only env vars that do nothing (HELIOS_MAX_PENDING_OPS etc.) so ops docs match code.
- [ ] **Secrets scrub (public repo!)**: rotate + purge the committed Discord webhook + ALPHA_REMOTE_SECRET
      (`scripts/nanochat-loop.sh`), ELEVENLABS key (`movies/`), dead GCP IPs. Add gitleaks-style check.
- **Gate G1 PASSED 2026-07-22:** 1,000-step, 6-layer Llama-form pilot (5.87M params, f32 Helios,
  no fallback, cooperative matrices disabled) completed with **ZERO non-finite loss/gradient events,
  ZERO NaN/spike skips**, and all 1,000 metric rows finite. Loss 8.3506→5.2346; final val 5.3624;
  median 4,686 tok/s; 905.8s total. CPU↔NVIDIA parity is 44/44. Evidence and command:
  `/mnt/donto-data/alpha-runs/g1-pilot-1000-20260722/RUN.md`. The old "2-7% NaN is normal" era is over.
  Later source trees are re-certified with `run_nvidia_gates.sh` (`1019b9b`), which rejects Vitest's
  exit-zero all-skipped state and emits proof only for vendor `0x10de` with all 46 current Helios
  assertions executed and passed.

### Stage 2 — Throughput: make 50-75M params affordable (≈$8 incl. one 6h soak)
Measured L4 history: 65K tok/s @1.85M → 30K @6.84M → ~4.9K @17.4M → **~1K @56M** (allocator-pressure
collapse). The root cause was that TS never passed `temporary=1`, bypassing the native slab pool and
turning every device tensor into an individual `vkAllocateMemory`. The current tree fixes that path:
- [x] Wire device-local slab: temporary output/intermediate buffers now use aligned, coalescing slab
      subranges with live-hole reuse and allocator telemetry (`f7730c6`, `32392a5`). NVIDIA reuse gates
      pass; the 100-step flagship-shape comparison improved steady throughput 3,322→3,790 tok/s (+14.1%).
      The six-hour boundedness proof passed below.
- [ ] Re-profile dispatch on the 3090/4090 (the dispatch-overhaul ring + batched dispatch exist; measure
      where the time actually goes at 16L/512d — kernel time vs submit vs GC).
- [x] Sweep `HELIOS_WG_SIZE` / pool caps per GPU. RTX 3090 profile is committed in `docs/RUNPOD.md`:
      WG=64, output-pool cap=512, cooperative matrices disabled. The exact profile passes 46/46 NVIDIA
      gates and sustains ~3.7–3.9K tok/s at the 57.69M flagship shape.
- **Gate G2 PASSED 2026-07-22:** ≥3,000 tok/s sustained (f32, flagship shape, block 1024) on a ≤$0.35/hr GPU, AND a 6-hour
  soak with zero allocator crashes and flat RSS/live-alloc curve.** Stretch: 8K tok/s.
  Budget math this gate protects: 1B tokens @3K tok/s ≈ 93 GPU-h ≈ **$20 on the 3090**. If the gate
  fails after honest effort: shrink flagship to ~35-40M (12L/448d) and/or cut token budget — decided
  then, in the ledger, not silently.
  Exact commit `aca9f97`, RTX 3090 at $0.22/hr: 5,400/5,400 finite rows, 88,473,600 tokens, 23,122
  monitored seconds (6h25m), throughput p10/median 3,721/3,832 tok/s, RSS 681–767MB with -4.73MB/1K
  slope, live allocations 1,073–1,157, Vulkan allocations 654–658, 34 constant temporary slabs, and
  zero free-range overflow. Loss mean fell 7.482→3.732; terminal train/val loss was 3.726/3.702. The
  full 692,528,815-byte checkpoint and every log are hash-sealed. Machine record:
  `/mnt/donto-data/alpha-runs/g2-soak-wg64-b16-5400-20260722/g2-analysis.json`; human record: adjacent
  `RUN.md`. The default ~60M architecture remains the flagship; no G2 shrink is warranted.

### Stage 3 — Modern architecture, Llama-shaped on purpose (box work + ≈$4 pilots)
The Llama-form implementation is complete on the current tree; the remaining gate work is the equal-token
100M-token comparison. The conversion kept exactly the set needed for zero-code `LlamaForCausalLM`:
- [x] **RoPE** (new op: fwd rotation + bwd inverse; cpu_ref + Helios kernel; applied to q/k post-sliceQkv;
      drop wpe). ~2-4 days per recon estimate.
- [x] **RMSNorm** (strict simplification of existing LayerNorm kernels; `normType` config).
- [x] **Tied embeddings** (`lmHead === wte`; shared-variable gradient accumulation parity-tested).
- [x] **Drop softCap** in the new arch (no Llama equivalent). If instability returns, the sanctioned
      fallback is **QK-RMSNorm + publish as Qwen3ForCausalLM** (also a stock zero-code arch) — decide by
      pilot, record in the ledger.
- [x] **Byte-level BPE** with the GPT-2 split regex (JS `/u` regex), 256-byte base + specials
      `<|user|>` `<|assistant|>` `<|end_of_text|>` atomic; byte-buffer decode; artifact schema v2;
      **tokenizer.json exporter** (ByteLevel pre-tokenizer + string-pair merges) proven equal on a 10K-doc
      round-trip vs `@huggingface/tokenizers`. Kills the OOV-silent-drop bug for real user input.
      Vocab: **12,288** (multiple of 256 for k-quants; tied, so embed cost is paid once).
- [x] Keep MHA (a valid Llama config: `num_key_value_heads == num_attention_heads`). **GQA: SKIP** —
      flash fwd+bwd kernels assume equal head counts; 3-5 days of kernel work for negligible gain ≤100M.
- [x] Parity tests (Stage 1 harness) extended to every new op BEFORE any paid run uses it.
- [x] Update `packages/inference` (CPU engine) for RoPE/RMSNorm/SwiGLU/tied so serving matches training
      rather than silently applying the former GELU-4x assumptions.
- [x] Equal-token pilot reproducibility: train/validation loaders use independent seeded streams (model
      parameter-count differences cannot perturb validation windows); packed/random/SFT loaders seek to
      checkpoint-consistent batch positions on resume. `run_g3_pilot.sh` records commit/data/tokenizer/
      parameter/token contracts; `analyze_g3_pair.ts` rejects non-finite, mismatched, or unaligned runs.
      Paid-pilot resume (`58fc691`) requires that exact contract, preserves+hashes any post-checkpoint
      metric tail, atomically realigns the append stream, and records a resume ledger.
- **Gate G3: 100M-token pilot of the new arch ≥ matches the old arch's loss curve at equal tokens/params,
  0 NaN steps, and a golden-token test: Alpha forward == exported-safetensors-in-transformers forward
  (top-1 agreement on 512 positions, fixed prompt) BEFORE the flagship run.** That last check is the
  from-scratch equivalent of "reproduce a pretrained checkpoint's logits" — it validates RoPE/RMSNorm/
  SwiGLU/export in one shot, without importing anyone's weights.
  **IN PROGRESS 2026-07-22:** exact certified source `c95f81b`; the Llama half started at 21:20:08 UTC
  as `g3-llama-100m-lr3e4-c95f81b-20260722`. Both 1.992GB data and tokenizer hashes match the sealed
  mounted inputs. A persistent 60-second guard mirrors metrics/logs/checkpoints and retains the latest
  three on each side. The one-time cache build produced 463,290,711 train + 51,536,242 validation tokens
  in 1,083.4s. The step-1,000 durability milestone is proven: 1,000/1,000 continuous finite rows;
  validation improved 5.5984302→5.0261204; host RSS 2,872MB; 34 temporary slabs; zero allocator
  overflow; explicit f32 posture. The full 692,528,815-byte ALPH checkpoint has identical remote/local
  SHA-256 `af862a5d…`, and its native-format header, tensor shapes, exact payload length, and all
  57,688,576 finite/nonzero parameters passed audit. The step-1,500 milestone is also proven:
  1,500/1,500 consecutive finite rows, validation 5.5984302→5.0261204→4.7217364, 3,864 tok/s median
  after warmup, 34 constant temporary slabs, and zero overflow. Remote/mounted metrics and the retained
  checkpoint were hash-identical. The exact three-shard flagship corpus was staged concurrently at low
  priority and all 5,976,889,749 remote bytes match the immutable manifest hashes. At step 2,000,
  validation improved again to 4.4397746 and the second 692,528,815-byte checkpoint passed remote/local
  hash parity plus a full native-format/finite-parameter audit. Its unchanged 5,284,184kB HWM proved
  checkpoint allocation reuse, while elevated post-save RSS exposed delayed collection of cloned AdamW
  buffers. A local post-save-GC + external-memory telemetry hardening passes TypeScript and 200/46 tests;
  deployment waits until the exact `c95f81b` architecture pair is complete. The Llama half completed
  normally at 6,104/6,104 rows and exactly 100,007,936 tokens. Canonical `summarizePilot` passed:
  57,688,576 params, 3,876 tok/s median, final/last-100 train loss 3.8499150/3.7737795, final held-out
  loss 3.7274671, last-three validation mean 3.7829017, 63 complete allocator samples, zero overflow.
  Terminal checkpoint 6,104 is a hash-mirrored/native-audited 692,528,815-byte ALPH file with every
  parameter finite/nonzero; the final guard safely retained exactly 5,000/6,000/6,104 and exited.
  The sequential GPT-2 control completed on the unchanged tree and exact same inputs/LR/schedule:
  6,104/6,104 finite/consecutive rows, exactly 100,007,936 tokens, 58,094,592 params (+0.704%),
  4,704 tok/s median, final/last-100 train loss 4.0688343/3.9916938, final held-out loss 3.9457434,
  63 complete allocator samples, and zero overflow. Its terminal 697,403,761-byte checkpoint is
  hash-mirrored/native-audited with every parameter finite/nonzero; the guard safely retained exactly
  5,000/6,000/6,104 and exited. **G3 PASS:** canonical pair analysis confirms matching contracts,
  Llama wins all 12 aligned validations, final advantage 0.2182763 and last-three mean advantage
  0.2302373, with zero overflow in both runs. Immutable report:
  `/mnt/donto-data/alpha-runs/g3-pair-analysis-c95f81b-20260723.json`.

### Stage 4 — Data (box only, $0 GPU)
Alpha has never pretrained on broad text — it went straight to chat data (SODA at 0.45 tokens/param).
That, not the framework, is half of why every prior run produced gibberish. Fix the pipeline:
- [x] **Pretrain corpus**: stream `HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled`
      (verified: 100B tokens, ODC-BY, streaming-safe, pre-shuffled) → clean text shards with
      `<|end_of_text|>` separators on `/mnt/donto-data/alpha-corpora/` → rsync to pod. Slice size set by
      G2 throughput (1B tokens ≈ 4GB text at ~4 chars/token; loader RAM = 4 bytes/token — fine ≤3B).
      **Ready:** 1,857,705 documents, 11.7GB text, ≈3.0B tokens in six shards; source parquets retained.
- [x] **Chat corpus (SFT)**: `HuggingFaceTB/smol-smoltalk` (460K convs, built FOR sub-1B models) as the
      backbone + `smoltalk2` `everyday-conversations_no_think` + `systemchats-30k_no_think` +
      OASST2 English best-ranked paths + ≤5% SODA seasoning → rendered to
      `<|user|> … <|assistant|> … <|end_of_text|>`. **Final:** 511,428/511,428 structurally clean,
      SODA 4.828%, SHA-256 `ffad0a37…`; the exact tokenizer bounded every row at 1,024 without cutting a
      response/EOS (whole trailing pairs only), and the all-row audit measured p50/p95/p99/max
      657/978/1,014/1,024 with zero over-bound. The real training mask passed on 1,032 rows spanning every
      source boundary. Reproduction and artifact paths: `docs/SFT_CORPUS.md`.
- [x] **Loss masking (assistant-only SFT)** — net-new, the single most important training-code change:
      `DataBatch.lossMask [B,T]`, masked cross-entropy (`sum(ce*mask)/max(sum(mask),1)`) in cpu_ref +
      Helios CE kernels + autograd, document-aware SFT batching (no packing v1), mask verified by
      decoding batches (user tokens 0, assistant tokens 1). Parity-tested (Stage 1 harness).
- [x] **Frozen eval set** (before flagship): per-corpus val shards; 100 fixed multi-turn chat prompts;
      200 closed-book sanity questions (from finewiki pages excluded from training); repetition/EOS/
      role-leak metrics; deterministic greedy sample suite at every checkpoint. No benchmark data in
      training mixes, ever. **Frozen at** `/mnt/donto-data/alpha-corpora/frozen-eval-v1`: 100 token-fit,
      source-balanced chats (49 held-out OASST2, 48 Magpie, 3 everyday; 84 multi-turn); 200 structured
      FineWiki questions; 500 held-out documents per premix source. The streaming Rust audit scanned
      1.918B pretrain + 205.0M final-SFT 13-grams and excluded 1,298/5,100 + 658/900
      contaminated candidates. Manifest hashes every source/output. `alpha eval-frozen` now scores greedy
      EOS/role leak/repetition plus QA exact/containment/F1; end-to-end smoke passed. Its v2 output and
      `analyze_frozen_eval_pair.ts` (`863427f`) hash/recompute both detailed suites, require exact
      base/chat steps and identical inputs/case order, bind both runs to canonical final-manifest
      chat/QA hashes, and enforce the machine-verifiable D3 threshold
      while explicitly reserving open-ended coherence for a separate semantic review.
- **Gate G4: PASS 2026-07-22.** Validation is decontaminated vs the exact final train text by 13-gram
  overlap; exact lengths are bounded; structural/mask checks are green; Alpha↔HF tokenizer parity is
  100/100 on frozen prompts and the 10K-document tokenizer export proof remains green.

### Stage 5 — Flagship runs (the big spend: ≈$25-35)
- **Shape (default)**: ~60M Llama-form — 16L / 512d / 8 heads (headDim 64) / SwiGLU ffn 1408 / RoPE /
  RMSNorm / tied / vocab 12,288 / block 1024 / f32. (Nearest proven ancestor: the 56M nanochat shape.)
  G2 decides final size; the ledger records any change.
- **Pretrain**: 1B tokens minimum (17 tokens/param — demo-grade but honest; stretch 2-3B if G2 lands
  ≥5K tok/s). AdamW β(0.9, 0.95), wd 0.1, grad-clip 1.0, **lr swept at 100M-token pilot scale over
  {1e-3, 2e-3, 3e-3}** (SmolLM2-135M used 3e-3; Alpha's old 3e-4 lore was tuned around bugs we've now
  fixed — re-derive, don't inherit), warmup 1-2%, cosine (WSD optional later; cosine is what the trainer
  has). `analyze_lr_sweep.ts` (`61c1edb`) rejects mismatched contracts and selects the lowest final-three
  aligned held-out-loss mean with deterministic tie-breaks. Batch ≈ 128-256K tokens/step via grad accum.
  Checkpoint on its independent cadence; **box-side puller** rsyncs
  every checkpoint off-pod (community pods are disposable; 5s SIGTERM on spot).
  The minimum-run input is the SHA-verified first-three-shard `flagship-1b-manifest.json` (~6GB/~1.5B
  estimated tokens): `28c6506` caches shards independently and reads them as one deterministic logical
  corpus, so 1B training tokens neither repeat one ~450M split nor exceed Node's giant-buffer limits.
  Cache files are keyed by exact tokenizer-artifact SHA and written by checked chunked I/O to an fsynced
  temporary followed by atomic rename (`45bfe60`); legacy/truncated/source-size-mismatched caches rebuild.
  `run_flagship_pretrain.sh` consumes the analyzer's hash-bound LR-selection report (not a hand-entered
  member of the sweep) and fixes the exact
  61,036-step / 1,000,013,824-token architecture, optimizer, eval, checkpoint, manifest, tokenizer,
  commit, and resume contract. `analyze_flagship_pretrain.ts` is the terminal fail-closed gate: it
  recomputes those bindings, requires every finite/consecutive metric and aligned validation, complete
  zero-overflow allocator telemetry, ≥3K tok/s p10/median, and native-audits every terminal parameter.
  **LR SWEEP PASSED 2026-07-24:** functional/source commit `e6d9430` rebuilt 19/19 on the RTX 3090 pod
  and passed the fail-closed NVIDIA gate 46/46 with zero skipped/failed/todo. The `1e-3` Llama
  candidate is complete: strict summary passed 6,104/6,104 consecutive finite rows and exactly
  100,007,936 tokens, median post-warmup throughput 3,892 tok/s, final train loss 3.6922, and
  final-three held-out-loss mean 3.6045400. All 63 allocator samples are complete with zero overflow;
  terminal checkpoint 6,104 is a 692,528,815-byte, hash-mirrored/native-audited ALPH file at
  `e43ce5a9…` with every parameter finite/nonzero. Its guard retained exactly 5,000/6,000/6,104,
  logged `final pull complete`, and exited. The second `2e-3` candidate is also complete: 6,104
  consecutive finite rows, exactly 100,007,936 tokens, median post-warmup throughput 3,843 tok/s,
  final train loss 3.7847, and final-three held-out-loss mean 3.6954683. All 63 allocator samples are
  complete with zero overflow. Its terminal checkpoint is a hash-mirrored/native-audited
  692,528,815-byte ALPH file at `ecb79332…`, with every parameter finite/nonzero; its guard retained
  exactly 5,000/6,000/6,104, completed the final pull, and exited. The third `3e-3` candidate also
  completed the identical contract: 6,104 finite/consecutive rows, exactly 100,007,936 tokens,
  3,862 tok/s median, final train loss 4.1647, final-three held-out-loss mean 4.1337789, 63 complete
  allocator samples/zero overflow, and a hash-mirrored/native-audited terminal checkpoint at
  `18cdcec8…`; its guard retained 5,000/6,000/6,104 and exited after final pull. The strict analyzer
  selected `1e-3`, ranking final-three means 3.6045400, 3.6954683, 4.1337789. Report:
  `/mnt/donto-data/alpha-runs/lr-sweep-analysis-e6d9430-20260724.json`, SHA-256 `10d39e47…`.
  Every sweep candidate stayed on `e6d9430`. Current-origin `e561f66` subsequently built 19/19 and
  passed the real RTX 3090 gate 46/46 with zero failed/skipped/todo. A four-cycle live-GPU proof then
  wrote four consecutive 692,528,809-byte checkpoints: each released 228 cloned AdamW buffers, ran GC,
  returned ArrayBuffers to the identical 2,705MB post-GC baseline, and independently passed an exact
  ALPH header/payload plus all-57,688,576-parameter finite/nonzero audit. The selected `1e-3` flagship
  launched at 10:51 UTC on exact source `e561f66`; its immutable contract binds 61,036 steps,
  1,000,013,824 tokens, selector `10d39e47…`, manifest `c7ecaf7d…`, and tokenizer `c310343a…`.
  All 5,976,889,749 source bytes hash-verified before startup; the two missing shard caches were built
  atomically, adding exactly 1,029,128,000 cached train/validation tokens. The first aligned held-out
  checkpoint gate passed 1,000/1,000 finite/consecutive rows and 16,384,000 tokens: train loss
  9.4982→4.8432, held-out loss 5.4226→4.8698 across steps 500/1,000, p10/median throughput after step
  50 3,730/3,862 tok/s, 11 complete allocator samples, 34 slabs, zero overflow, and bounded
  7,804–8,960MB RSS. The save released all 228 cloned optimizer buffers with GC. Remote/mounted
  checkpoint 1,000 is a hash-identical/native-audited 692,528,815-byte ALPH file at `93ddc593…` with
  all parameters finite/nonzero; checkpoint-1,000 metrics match at `bc616a21…`. The step-1,500
  held-out gate then passed 1,500 finite rows/24,576,000 tokens, train/held-out loss 4.4025/4.4596,
  validation improvement 0.4102, p10/median 3,725/3,856 tok/s, 16 allocator samples, 34 slabs, and
  zero overflow. Remote/mounted metrics match at `a3860b8b…`; all 500 post-checkpoint rows held
  ArrayBuffers exactly at 6,632MB and RSS within 7,793–7,869MB. The checkpoint-2,000 gate then
  passed 2,000 finite rows/32,768,000 tokens, train/held-out loss 4.2562/4.2743, another 0.1853
  validation improvement, p10/median 3,723/3,849 tok/s, 21 allocator samples, 34 slabs, and zero
  overflow. Checkpoint `7f54b34a…` and metrics `01a31962…` are hash-identical remote/mounted and the
  native audit passed all parameters finite/nonzero. The second save released all 228 optimizer
  buffers and returned ArrayBuffers 7,072→6,631MB, proving there is no per-checkpoint accumulation.
  The step-2,500 held-out gate then passed 2,500 finite rows/40,960,000 tokens, train/held-out loss
  4.0449/4.1624, another 0.1119 validation improvement, p10/median 3,722/3,850 tok/s, 26 allocator
  samples, 34 slabs, and zero overflow. Metrics are hash-identical remote/mounted at `44a82dea…`;
  all 500 rows after checkpoint 2,000 held ArrayBuffers exactly at 6,632MB and RSS within
  7,883–7,942MB. The checkpoint-3,000 gate then passed 3,000 finite rows/49,152,000 tokens,
  train/held-out loss 4.0256/4.0843, another 0.0781 validation improvement, p10/median
  3,726/3,852 tok/s, 31 allocator samples, 34 slabs, and zero overflow. Checkpoint `a2a56b81…` and
  exact metrics prefix `e0139d26…` are hash-identical remote/mounted; the native audit passed all
  57,688,576 parameters finite/nonzero. The third save again released all 228 buffers and returned
  ArrayBuffers 7,072→6,631MB. The step-3,500 held-out gate then passed 3,500 finite rows/57,344,000
  tokens, train/held-out loss 3.8251/3.9699, another 0.1144 validation improvement, p10/median
  3,723/3,850 tok/s, 36 allocator samples, 34 slabs, and zero overflow. Metrics are hash-identical
  remote/mounted at `6a3f69cf…`; all 500 rows after checkpoint 3,000 held ArrayBuffers exactly at
  6,632MB and RSS within 7,885–7,943MB. The checkpoint-4,000 gate then passed 4,000 finite rows/
  65,536,000 tokens, train/held-out loss 4.0469/3.8976, another 0.0723 validation improvement,
  p10/median 3,724/3,850 tok/s, 41 allocator samples, 34 slabs, and zero overflow. Checkpoint
  `25b061b5…` and metrics `79c4a1b9…` are hash-identical remote/mounted; the native audit passed all
  57,688,576 parameters finite/nonzero. The fourth save again released all 228 buffers and returned
  ArrayBuffers 7,072→6,631MB. The first live prune then safely removed checkpoint 1,000 remotely only
  after its mounted size/SHA match, followed by ledgered local removal of the identical `93ddc593…`
  artifact. The step-4,500 held-out gate then passed 4,500 finite rows/73,728,000 tokens,
  train/held-out loss 3.8755/3.8603, another 0.0373 validation improvement, p10/median
  3,724/3,850 tok/s, 46 allocator samples, 34 slabs, and zero overflow. Metrics are hash-identical
  remote/mounted at `654595d1…`; all 500 rows after checkpoint 4,000 held ArrayBuffers within
  6,631–6,632MB and RSS within 7,868–7,933MB. The checkpoint-5,000 gate then passed 5,000 finite
  rows/81,920,000 tokens, train/held-out loss 3.8075/3.7961, another 0.0642 validation improvement,
  p10/median 3,723/3,850 tok/s, 51 allocator samples, 34 slabs, and zero overflow. Checkpoint
  `b9851894…` and metrics `34a5e893…` are hash-identical remote/mounted; the native audit passed all
  57,688,576 parameters finite/nonzero. The fifth save again released all 228 buffers and returned
  ArrayBuffers 7,072→6,631MB. The second live prune safely removed checkpoint 2,000 remotely only
  after mounted size/SHA proof, followed by ledgered local removal of the same `7f54b34a…` artifact.
  The step-5,500 gate then passed 5,500 finite rows/90,112,000 tokens. Train loss was 3.7954;
  held-out loss 3.8107 was a small +0.0146 wobble from step 5,000, while remaining 1.6119 below step
  500. P10/median throughput was 3,723/3,850 tok/s; all 56 allocator samples report 34 slabs and zero
  overflow. Metrics are hash-identical remote/mounted at `bbc5e153…`; every post-checkpoint row held
  ArrayBuffers exactly at 6,632MB and RSS within 7,860–7,931MB. The checkpoint-6,000 gate then
  passed 6,000 finite rows/98,304,000 tokens: train/held-out loss 3.5498/3.7055, recovering the
  step-5,500 wobble with a new-best 0.1051 validation improvement, p10/median 3,721/3,849 tok/s,
  61 allocator samples, 34 slabs, and zero overflow. Checkpoint `6b171970…` and metrics `616f5385…`
  are hash-identical remote/mounted; the native audit passed all 57,688,576 parameters
  finite/nonzero. The sixth save again released all 228 buffers and returned ArrayBuffers
  7,072→6,631MB. The third live prune safely removed checkpoint 3,000 remotely only after mounted
  size/SHA proof, followed by ledgered local removal of the same `a2a56b81…` artifact. Training
  resumed through step 6,050; both sides retain exactly 4,000/5,000/6,000. The step-6,500 gate then
  passed 6,500 finite rows/106,496,000 tokens. Train loss was 3.6452; held-out loss 3.7176 was a small
  +0.0120 wobble from step 6,000 while remaining 0.0931 below step 5,500. P10/median throughput was
  3,723/3,850 tok/s; all 66 allocator samples report 34 slabs and zero overflow. Metrics are
  hash-identical remote/mounted at `9d1ff974…`; every post-checkpoint row held ArrayBuffers exactly
  at 6,632MB and RSS within 7,865–7,934MB. The checkpoint-7,000 gate then passed 7,000 finite rows/
  114,688,000 tokens: train/held-out loss 3.5480/3.7157, a 0.0019 validation improvement from step
  6,500 and only 0.0101 above the step-6,000 best. P10/median throughput was 3,722/3,849 tok/s; all
  71 allocator samples report 34 slabs and zero overflow. Checkpoint `b26165fd…` and metrics
  `4c835219…` are hash-identical remote/mounted; the native audit passed all 57,688,576 parameters
  finite/nonzero. The seventh save again released all 228 buffers and returned ArrayBuffers
  7,072→6,631MB. The fourth live prune safely removed checkpoint 4,000 remotely only after mounted
  size/SHA proof, followed by ledgered local removal of the same `25b061b5…` artifact. Training
  resumed through step 7,025; both sides retain exactly 5,000/6,000/7,000. The step-7,500 gate then
  passed 7,500 finite rows/122,880,000 tokens: train/held-out loss 3.6169/3.6547, a new best by
  0.0610 from step 7,000 and 0.0509 below the prior best at step 6,000. P10/median throughput was
  3,723/3,850 tok/s; all 76 allocator samples report 34 slabs and zero overflow. Metrics are
  hash-identical remote/mounted at `0dc719fc…`; every post-checkpoint row held ArrayBuffers exactly
  at 6,632MB and RSS within 7,854–7,936MB. The checkpoint-8,000 gate then passed 8,000 finite rows/
  131,072,000 tokens: train/held-out loss 3.5756/3.6440, another new best by 0.0106 from step 7,500
  and 0.0615 below step 6,000. P10/median throughput was 3,724/3,850 tok/s; all 81 allocator samples
  report 34 slabs and zero overflow. Checkpoint `e7658b21…` and metrics `8b1679e0…` are hash-identical
  remote/mounted; the native audit passed all 57,688,576 parameters finite/nonzero. The eighth save
  again released all 228 buffers and returned ArrayBuffers 7,072→6,631MB. The fifth live prune
  safely removed checkpoint 5,000 remotely only after mounted size/SHA proof, followed by ledgered
  local removal of the same `b9851894…` artifact. Training resumed through step 8,050; both sides
  retain exactly 6,000/7,000/8,000. The step-8,500 gate then passed 8,500 finite rows/139,264,000
  tokens. Train loss was 3.4844; held-out loss 3.7603 was a 0.1163 wobble from the step-8,000 best
  while remaining 0.0504 below step 5,500 and 1.6623 below step 500. P10/median throughput was
  3,725/3,849 tok/s; all 86 allocator samples report 34 slabs and zero overflow. Metrics are
  hash-identical remote/mounted at `c301b0b5…`; every post-checkpoint row held ArrayBuffers exactly
  at 6,632MB and RSS within 7,871–7,937MB. Overnight, every gate through checkpoint 12,000 passed:
  held-out loss at steps 9,000/9,500/10,000/10,500/11,000/11,500/12,000 was 3.6261/3.6613/3.6328/
  3.6641/3.6619/3.5723/3.4737, decisively resolving the step-8,500 wobble and ending at a new best.
  All 12,000 rows are finite/consecutive and cover 196,608,000 tokens; p10/median throughput is
  3,721/3,848 tok/s, and all 121 allocator samples report 34 slabs and zero overflow. Checkpoints
  9,000/10,000/11,000/12,000 at `7ce876e5…`/`9352634d…`/`ada7bf46…`/`61eccbe3…` were each
  hash-mirrored and native-audited with all 57,688,576 parameters finite/nonzero; the exact
  12,000-row metrics prefix matches at `5998c8cf…`. The guard safely pruned checkpoints 6,000/
  7,000/8,000/9,000 only after mounted proof and ledgered local deletion, leaving exactly
  10,000/11,000/12,000 on both sides. All four saves released 228 buffers; the step-12,000 immediate
  6,694MB ArrayBuffers reading settled to the usual 6,632MB by step 12,050. Training is live through
  step 12,050. The next unattended interval also passed through checkpoint 22,000: all 22,000 rows
  are finite/consecutive and cover 360,448,000 tokens (36.0443%); p10/median throughput is
  3,734/3,865 tok/s; all 221 allocator samples have the required cadence, exactly 34 slabs, and zero
  overflow. Held-out loss across 12,500–22,000 remained bounded and reached a new best 3.4007978 at
  step 20,000; step 22,000 closed at 3.4330427. The exact 22,000-row prefix matches at `d3dc3886…`.
  Checkpoints 19,000/20,000/21,000/22,000 were hash-mirrored and independently native-audited at
  `c70d86af…`/`bc64cec9…`/`dafaddf7…`/`2b4d4df5…`, with all 57,688,576 parameters finite/nonzero.
  The audit report for 19,000 was preserved before bounded retention removed its checkpoint. The
  guard's mirror+ledger proof covers the earlier unattended checkpoints; 13,000–18,000 were already
  pruned before a retrospective native scan and are not claimed as native-audited. Both sides retain
  exactly 20,000/21,000/22,000. Every post-12,000 metric row holds ArrayBuffers within
  6,631–6,632MB and RSS within 7,851–7,944MB; all subsequent saves released 228 buffers directly to
  6,631MB. Training resumed beyond 22,000 with the RTX 3090 at 100% utilization. Balance was
  `$47.7200922733` at approximately 13:36 UTC. The complete SFT input contract and canonical frozen
  eval set were then staged under `/runpod/data/alpha-sft-v2` and `/runpod/data/frozen-eval-v1` at
  low I/O priority; all eight remote SHA-256 values match the mounted corpus/audit/tokenizer/manifest/
  chat/QA artifacts exactly, and the trainer continued advancing during verification. Step 22,500
  then passed 22,500 finite/consecutive rows and 368,640,000 tokens (36.8635%), with p10/median
  3,735/3,865 tok/s, 226 complete allocator samples, 34 slabs, zero overflow, and exact metrics hash
  `d25d85ff…`. Held-out loss 3.5532966 is a +0.1202539 one-gate wobble from step 22,000; it is on
  watch rather than prompting intervention because all numeric/memory/allocator invariants remain
  green and the earlier comparable step-8,500 wobble recovered. Checkpoint 23,000 is the next
  discriminator. Balance was `$47.5512524306` at approximately 14:09 UTC. Checkpoint 23,000 then
  resolved the wobble: held-out loss recovered to 3.4372567, only 0.0042140 above step 22,000.
  All 23,000 rows are finite/consecutive and cover 376,832,000 tokens (37.6827%); p10/median is
  3,734/3,864 tok/s; all 231 allocator samples report 34 slabs/zero overflow. Exact metrics
  `32da13ab…` and checkpoint `746e14f4…` match remote/mounted; the native audit passed all
  57,688,576 parameters finite/nonzero, and the save released 228 buffers to 6,631MB. Safe retention
  removed checkpoint 20,000 only after proof and now holds exactly 21,000/22,000/23,000 on both
  sides. Training resumed through 23,025; balance was `$47.3824188825` at approximately 14:46 UTC.
  Step 23,500 also passed: 23,500 finite/consecutive rows cover 385,024,000 tokens (38.5019%), with
  p10/median 3,732/3,862 tok/s, 236 complete allocator samples, 34 slabs, zero overflow, and exact
  remote/mounted metrics prefix `82f84baa…`. Held-out loss 3.5132058 is +0.0759491 from step 23,000
  but 0.0400908 better than step 22,500; with advancing five-batch windows and every numeric/memory/
  allocator invariant green, this remains normal variance pending checkpoint 24,000. Balance was
  `$47.2135497066` at approximately 15:21 UTC.
  Checkpoint 24,000 then passed with 24,000 finite/consecutive rows and 393,216,000 tokens
  (39.3211%), p10/median 3,731/3,861 tok/s, 241 complete allocator samples, 34 slabs, and zero
  overflow. Held-out loss improved to 3.4923591. Exact metrics `66e75c19…` and the 692,528,817-byte
  checkpoint `1c80ee85…` match remote/mounted; the native audit passed all 57,688,576 parameters
  finite/nonzero, and the save released 228 buffers to 6,631MB. Safe retention removed checkpoint
  21,000 only after mirror proof and now holds exactly 22,000/23,000/24,000 on both sides. The RTX
  3090 returned to 100% utilization; balance was `$47.0205838101` at approximately 16:00 UTC.
  Step 24,500 also passed with 24,500 finite/consecutive rows and 401,408,000 tokens (40.1402%),
  p10/median 3,730/3,860 tok/s, 246 complete allocator samples, 34 slabs, and zero overflow. Held-out
  loss was effectively flat at 3.4938740 (+0.0015149 from 24,000), while the exact remote/mounted
  metrics prefix matches at `f7c2f6a6…`. Post-24k ArrayBuffers remained exactly 6,632MB and RSS
  7,878–7,937MB. Balance was `$46.8517209286` at approximately 16:34 UTC.
  Checkpoint 25,000 then passed with 25,000 finite/consecutive rows and 409,600,000 tokens (40.9594%),
  p10/median 3,730/3,859 tok/s, 251 complete allocator samples, 34 slabs, and zero overflow. Held-out
  loss improved to 3.4471820. Exact metrics `78e73346…` and the 692,528,817-byte checkpoint
  `8a86ca42…` match remote/mounted; the native audit passed all 57,688,576 parameters finite/nonzero,
  and the save released 228 buffers to 6,631MB. Safe retention removed checkpoint 22,000 only after
  mirror proof and now holds exactly 23,000/24,000/25,000 on both sides. Balance was
  `$46.6829211138` at approximately 17:10 UTC.
  Step 25,500 also passed with 25,500 finite/consecutive rows and 417,792,000 tokens (41.7786%),
  p10/median 3,729/3,859 tok/s, 256 complete allocator samples, 34 slabs, and zero overflow. Held-out
  loss was effectively flat at 3.4464590, slightly better by 0.0007231 from 25,000. Exact remote/
  mounted metrics match at `b8dcc21a…`; post-25k ArrayBuffers stayed exactly 6,632MB and RSS within
  7,857–7,937MB. Balance was `$46.5140116656` at approximately 17:45 UTC.
  Checkpoint 26,000 then passed with 26,000 finite/consecutive rows and 425,984,000 tokens (42.5978%),
  p10/median 3,728/3,858 tok/s, 261 complete allocator samples, 34 slabs, and zero overflow. Held-out
  loss improved to 3.4225069, only +0.0217091 from the run best. Exact metrics `c4222263…` and the
  692,528,817-byte checkpoint `28b0050b…` match remote/mounted; the native audit passed all
  57,688,576 parameters finite/nonzero, and the save released 228 buffers to 6,631MB. Safe retention
  removed checkpoint 23,000 only after mirror proof and now holds exactly 24,000/25,000/26,000 on
  both sides. Training resumed through 26,025; balance was `$46.3211066359` at approximately 18:23 UTC.
  Step 26,500 then established a new run-best held-out loss of 3.3790116, improving 0.0217862 from
  the prior step-20,000 best. All 26,500 rows are finite/consecutive and cover 434,176,000 tokens
  (43.4170%); p10/median is 3,728/3,858 tok/s; all 266 allocator samples report 34 slabs/zero
  overflow. Exact remote/mounted metrics match at `822c37d3…`; post-26k ArrayBuffers stayed exactly
  6,632MB and RSS within 7,854–7,930MB. Balance was `$46.1522706432` at approximately 18:57 UTC.
  Checkpoint 27,000 then extended the run-best held-out loss to 3.3680425, improving 0.0109691 from
  step 26,500 and 0.0327553 from the former step-20,000 best. All 27,000 rows are finite/consecutive
  and cover 442,368,000 tokens (44.2362%); p10/median is 3,728/3,858 tok/s; all 271 allocator samples
  report 34 slabs/zero overflow. Exact metrics `b1fe1b30…` and checkpoint `972902b7…` match remote/
  mounted; the native audit passed all 57,688,576 parameters finite/nonzero, and the save released
  228 buffers to 6,631MB. Safe retention removed checkpoint 24,000 only after proof and now holds
  exactly 25,000/26,000/27,000. Training resumed through 27,050; balance was `$45.983374884`.
  Step 27,500 then passed with 27,500 finite/consecutive rows and 450,560,000 tokens (45.0554%);
  p10/median is 3,728/3,858 tok/s; all 276 allocator samples report 34 slabs/zero overflow. Held-out
  loss 3.3710784 is only +0.0030359 from the step-27,000 run best and remains better than every
  earlier gate. Exact remote/mounted metrics match at `4ca7f954…`; post-27k ArrayBuffers stayed
  exactly 6,632MB and RSS within 7,861–7,931MB. Training resumed through 27,550; balance was
  `$45.8145234914`.
  Checkpoint 28,000 then passed with 28,000 finite/consecutive rows and 458,752,000 tokens (45.8746%),
  p10/median 3,728/3,858 tok/s, all 281 allocator samples present, 34 slabs, and zero overflow.
  Held-out loss 3.4247354 is +0.0566929 from the 27,000 best but only +0.0022285 from checkpoint
  26,000. Exact metrics `a3bb4acf…` and checkpoint `b9f80989…` match remote/mounted; the native scan
  passed all 57,688,576 parameters finite/nonzero. The save released 228 buffers and training
  returned to the 6,632MB ArrayBuffers baseline at step 28,001. Safe retention removed checkpoint
  25,000 only after proof and now holds exactly 26,000/27,000/28,000. Training resumed through
  28,050; balance was `$45.6456990488`.
  Before the interruption, step 28,500 had passed with 28,500 finite/consecutive rows and
  466,944,000 tokens (46.6938%);
  p10/median is 3,728/3,857 tok/s; all 286 allocator samples report 34 slabs/zero overflow. Held-out
  loss improved 0.0300772 from checkpoint 28,000 to 3.3946582 and is only +0.0266157 from the
  step-27,000 run best. Exact remote/mounted metrics match at `d4764198…`; post-28k ArrayBuffers
  stayed exactly 6,632MB and RSS within 7,856–7,935MB. Training resumed through 28,525; balance was
  `$45.476881634`.
  RunPod then unexpectedly marked the original pod `Exited by user` at 21:49:25 UTC after its exact
  28,900-row metrics prefix had reached the mounted mirror but before checkpoint 29,000. No stop was
  issued by this session and guard auto-termination was disabled. The abandoned 28,001–28,900 tail
  remains preserved at SHA-256 `bec96f18…`; it is not counted in the canonical continuation.
  Recovery2 pod `gp4m6s8m06bhen` independently passed 46/46 NVIDIA gates, restored the exact audited
  28,000 checkpoint `b9f80989…`, verified all 5,976,889,749 corpus bytes, rebuilt all six token
  caches, ledgered the metrics truncation, and resumed on exact source `e561f66`. Canonical steps
  28,001–28,050 are consecutive/finite with mean loss 3.3464976, mean grad norm 0.2271370, and median
  4,000 tok/s; step 28,100 reports 34 slabs/zero overflow. The canonical replay then re-passed the
  full step-28,500 gate: 28,500 finite/consecutive rows, 466,944,000 tokens (46.6938%), p10/median
  3,728.9936/3,858.6430 tok/s, all 286 allocator samples present, 34 slabs, and zero overflow.
  Train/held-out loss was 3.2907429/3.3982231; held-out is only 0.0035649 above the abandoned
  trajectory's corresponding gate. Exact remote/mounted metrics match at SHA-256 `9a9edd57…`.
  After the one-time post-tokenization residue, steps 28,101–28,500 held RSS within 7,936–7,948MB
  and ArrayBuffers within 6,995–6,996MB. The replacement GPU remained at 100% and `/runpod` had
  8.7GB free. Only after the step-28,100 proof, the stopped original pod was deleted and the
  hash-verified redundant transfer archives were removed. The live guard now has a 1,800-second
  verified-metric stall limit and pod-scoped auto-termination. Balance was `$44.7557173066`.
  Checkpoint 29,000 then established the first new durable point beyond the failed host: 29,000
  finite/consecutive rows, 475,136,000 tokens (47.5129%), p10/median 3,729.7739/3,859.2590 tok/s,
  all 291 allocator samples, 34 slabs, and zero overflow. Train/held-out loss was
  3.2767162/3.4017447, only +0.0035215 from the canonical recovery step-28,500 gate. Exact metrics
  `5a1e0af4…` and the 692,528,817-byte checkpoint `2e66f8d3…` match remote/mounted; the independent
  native audit `a977b8aa…` passed all 114 parameter tensors / 57,688,576 elements finite and
  nonzero. Safe retention ledgered and removed local checkpoint 26,000, leaving 27,000/28,000/29,000
  locally and 28,000/29,000 remotely. The last pre-save 500 rows held ArrayBuffers exactly 6,996MB
  and RSS exactly 7,948MB. After the save released 228 optimizer buffers, steps 29,001–29,050
  settled at 7,292MB ArrayBuffers, +296MB over the recovery plateau; RAM remains safe, but checkpoint
  30,000 must distinguish a one-time resume/save residue from per-save accumulation. Training resumed
  through 29,050 at 100% GPU; balance was `$44.5628176547`.
  Step 29,500 then passed with 29,500 finite/consecutive rows and 483,328,000 tokens (48.3321%);
  p10/median was 3,730.4182/3,859.8298 tok/s and all 296 allocator samples reported 34 slabs/zero
  overflow. Train/held-out loss was 3.2149160/3.4070656, only +0.0053210 from checkpoint 29,000.
  Exact remote/mounted metrics match at SHA-256 `07b03e0c…`. Across steps 29,001–29,500,
  ArrayBuffers stayed exactly 7,292MB and RSS stayed within 8,179–8,258MB, proving the retained
  checkpoint block is not a per-training-step leak. Checkpoint 30,000 remains the per-save
  accumulation discriminator. Balance was `$44.4181244601`.
  Checkpoint 30,000 then passed and resolved the live-buffer discriminator: 30,000 finite/consecutive
  rows cover 491,520,000 tokens (49.1513%); p10/median was 3,731.2399/3,860.6254 tok/s and all 301
  allocator samples reported 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.3288895/0.2294206. Train/held-out loss was 3.1875825/3.3639263, a new run best by 0.0041162 over
  step 27,000. Exact 30,000-row metrics `f4a39944…` and the 692,528,817-byte checkpoint
  `1625c7d6…` match remote/mounted; native audit `6ec3b0ff…` passed all 114 tensors / 57,688,576
  elements finite and nonzero. After the save released all 228 clones, steps 30,001–30,050 returned
  to exactly 7,292MB ArrayBuffers, proving the +296MB recovery increment did not repeat per save.
  RSS settled 8,471–8,472MB (+~220MB from immediately pre-save) without live external-buffer growth,
  so it remains under observation at checkpoint 31,000. Safe retention is exactly 28k/29k/30k on
  both sides. Balance was `$44.2492298008`; mounted disk had 85GB free.
  Step 30,500 then passed with another held-out run best: 30,500 finite/consecutive rows cover
  499,712,000 tokens (49.9705%); p10/median was 3,731.9190/3,861.3858 tok/s and all 306 allocator
  samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.3303287/0.2359238 and held ArrayBuffers/external memory exactly 7,292/7,294MB, with RSS
  8,417–8,490MB. Train/held-out loss was 3.2476387/3.3596032, improving 0.0043231 from checkpoint
  30,000. Exact remote/mounted metrics matched at `c9efd9ed…`; guard remained active/zero-restart.
  Balance was `$43.9518146859`. Total account burn rose to `$0.75/hr` because an unrelated Wajarri
  A40 pod was running; Alpha remained scoped to its `$0.22/hr` RTX 3090 and no unrelated pod was
  touched.
  Checkpoint 33,000 then passed and the held-out wobble recovered: 33,000 finite/consecutive rows
  cover 540,672,000 tokens (54.0665%); p10/median was 3,736.2691/3,866.5968 tok/s and all 331
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.2952673/0.2418966 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,465–8,538MB. Train/held-out loss was 3.2223387/3.3299192, improving 0.0209901 from step 32,500
  and 0.0051561 from checkpoint 32,000; it remained 0.0406322 above the sharp step-31,500 best.
  Exact metrics `addc830f…` and the 692,528,817-byte checkpoint `000d1d09…` match remote/mounted;
  native audit `9893f0c0…` passed all 114 tensors / 57,688,576 elements finite and nonzero. Post-save
  steps 33,001–33,050 held the exact 7,292/7,294MB live-buffer baseline and the 8,528MB pre-save RSS,
  adding no new plateau. Retention was 31k/32k/33k both sides. Balance was `$42.523076967`, total
  burn `$0.303/hr`, mounted disk 82GB free.
  Step 33,500 then passed: 33,500 finite/consecutive rows cover 548,864,000 tokens (54.8856%);
  p10/median was 3,737.3560/3,867.5338 tok/s and all 336 allocator samples reported exactly 34
  slabs/zero overflow. The last 500 rows averaged loss/gradient norm 3.2911672/0.2427404 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,469–8,541MB. Train/held-out loss was
  3.1827683/3.3272114, improving 0.0027078 from checkpoint 33,000 and remaining 0.0379244 above the
  sharp step-31,500 best. Exact remote/mounted metrics matched at `20feca37…`; the guard remained
  active/zero-restart. Balance was `$42.35422643`, only Alpha was running, total burn was
  `$0.303/hr`, and mounted disk had 83GB free.
  Checkpoint 34,000 then passed: 34,000 finite/consecutive rows cover 557,056,000 tokens (55.7048%);
  p10/median was 3,738.1565/3,868.6421 tok/s and all 341 allocator samples reported exactly 34
  slabs/zero overflow. The last 500 rows averaged loss/gradient norm 3.2653971/0.2541467 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,465–8,539MB. Train/held-out loss was
  3.3614364/3.3465386, a +0.0193273 wobble from step 33,500 while remaining 0.0043707 better than
  step 32,500. Exact metrics `87ef124c…` and the 692,528,817-byte checkpoint `2d63169b…` matched
  remote/mounted; native audit `3102c2b0…` passed all 114 tensors / 57,688,576 elements finite and
  nonzero. Post-save steps 34,001–34,050 returned to exactly 7,292/7,294MB buffers and RSS 8,538MB.
  Retention was 32k/33k/34k both sides. Balance was `$42.185329143`, only Alpha was running, total
  burn was `$0.303/hr`, and mounted disk had 82GB free.
  Step 34,500 then passed all hard invariants with a held-out wobble on watch: 34,500
  finite/consecutive rows cover 565,248,000 tokens (56.5240%); p10/median was
  3,739.0630/3,869.5629 tok/s and all 346 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2622366/0.2468041 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,467–8,540MB. Train/held-out loss was
  3.3501410/3.3995402, a +0.0530016 wobble from checkpoint 34,000 and +0.1102532 above the sharp
  step-31,500 best. This is comparable to earlier recovered five-batch variance, so checkpoint
  35,000 is the discriminator and no intervention is justified from one read. Exact remote/mounted
  metrics matched at `9eb80597…`; guard remained active/zero-restart. Balance was `$41.8946601577`;
  total burn was `$0.75/hr` because unrelated Wajarri pod `2q7ky3hpzbsw17` was running at
  `$0.44/hr`. Alpha remained `$0.22/hr` and the unrelated pod was not touched.
  Checkpoint 35,000 then passed and decisively resolved the wobble to a new run best: 35,000
  finite/consecutive rows cover 573,440,000 tokens (57.3432%); p10/median was
  3,739.8840/3,870.6404 tok/s and all 351 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.3145007/0.2491159 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,467–8,539MB. Train/held-out loss was
  3.3514276/3.2819459, improving 0.1175943 from step 34,500 and setting a new run best by 0.0073411
  over step 31,500. Exact metrics `c4144895…` and the 692,528,817-byte checkpoint `df9dc23a…`
  matched remote/mounted; native audit `ce6e46a3…` passed all 114 tensors / 57,688,576 elements
  finite and nonzero. Post-save steps 35,001–35,050 returned to exactly 7,292/7,294MB buffers and
  RSS 8,539MB. Retention was 33k/34k/35k both sides. Balance was `$41.6485739058`, only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 78GB free.
  Step 35,500 then passed all hard gates with five-batch variance on watch: 35,500
  finite/consecutive rows cover 581,632,000 tokens (58.1624%); p10/median was
  3,740.7007/3,871.8695 tok/s and all 356 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.3062677/0.2554765 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,464–8,541MB. Train/held-out loss was
  3.2625861/3.3820238: +0.1000779 from the unusually strong checkpoint-35,000 best, but already
  0.0175164 better than the prior 34,500 wobble. Exact remote/mounted metrics matched at
  `f31899c0…`; guard remained active/zero-restart. Balance was `$41.458165665`; total burn was
  `$0.75/hr` because unrelated Wajarri pod `9u5z7t9uv6e8ac` was running at `$0.44/hr`. Alpha
  remained `$0.22/hr` and the unrelated pod was not touched.
  Checkpoint 36,000 then passed every hard gate while elevated validation persisted: 36,000
  finite/consecutive rows cover 589,824,000 tokens (58.9816%); p10/median was
  3,741.3099/3,873.2473 tok/s and all 361 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2717362/0.2514415 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,465–8,540MB. Train/held-out loss was
  3.2520704/3.3931745: +0.0111507 from step 35,500 and +0.1112286 from the sharp 35,000 best. Two
  elevated windows are explicit, but they remain within established oscillation and no hard stop
  condition fired. Exact metrics `8fb078fa…` and the 692,528,817-byte checkpoint `696a20f8…`
  matched remote/mounted; native audit `b5321d32…` passed all 114 tensors / 57,688,576 elements
  finite and nonzero. Post-save steps 36,001–36,050 returned to exactly 7,292/7,294MB buffers and
  RSS 8,540MB. Retention was 34k/35k/36k both sides. Balance was `$41.0687693316`, only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 76GB free.
  Step 36,500 then passed and recovered the elevated validation trend: 36,500 finite/consecutive rows
  cover 598,016,000 tokens (59.8008%); p10/median was 3,741.9697/3,874.4000 tok/s and all 366
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.2918009/0.2540529 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,467–8,550MB. Train/held-out loss was 3.3547421/3.3283298, improving 0.0648447 from checkpoint
  36,000 and 0.0536940 from step 35,500; it remained 0.0463839 above the sharp 35,000 best. Exact
  remote/mounted metrics matched at `747f7b02…`; guard remained active/zero-restart. Balance was
  `$40.9240139261`, only Alpha was running, total burn was `$0.303/hr`, and mounted disk had 74GB
  free.
  Checkpoint 37,000 then passed with a new run best and crossed 60%: 37,000 finite/consecutive rows
  cover 606,208,000 tokens (60.6200%); p10/median was 3,742.6444/3,875.4751 tok/s and all 371
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.2631735/0.2608119 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,528–8,539MB. Train/held-out loss was 3.2379632/3.2644020, improving 0.0639278 from step 36,500
  and setting a new run best by 0.0175439 over checkpoint 35,000. Exact metrics `e0e49b59…` and the
  692,528,817-byte checkpoint `5fddd499…` matched remote/mounted; native audit `a8419c04…` passed
  all 114 tensors / 57,688,576 elements finite and nonzero. Post-save steps 37,001–37,050 returned
  to exactly 7,292/7,294MB buffers and RSS 8,540MB. Retention was 35k/36k/37k both sides. Balance
  was `$40.5897592151`; total burn was `$0.75/hr` because unrelated Wajarri pod `2d55zbgwjg13ta`
  was running at `$0.44/hr`. Alpha remained `$0.22/hr`, and mounted disk had 74GB free.
  Step 37,500 then passed while remaining near that new best: 37,500 finite/consecutive rows cover
  614,400,000 tokens (61.4392%); p10/median was 3,743.6359/3,876.5917 tok/s and all 376 allocator
  samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2731337/0.2579882 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,465–8,542MB. Train/held-out loss was 3.2774620/3.2806316, only 0.0162296 above the step-37,000
  run best. Exact remote/mounted metrics matched at `4182bd2a…`; the guard remained active with
  zero restarts. Balance was `$40.2212785873`; total burn was `$0.75/hr` because unrelated Wajarri
  pod `2d55zbgwjg13ta` remained running at `$0.44/hr`. Alpha remained `$0.22/hr`, and mounted disk
  had 73GB free.
  Checkpoint 38,000 then passed with a slight validation improvement and a clean save/memory gate:
  38,000 finite/consecutive rows cover 622,592,000 tokens (62.2583%); p10/median was
  3,744.5871/3,877.7341 tok/s and all 381 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2679434/0.2623056 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,540–8,541MB. Train/held-out loss was
  3.3129361/3.2791747, improving 0.0014569 from step 37,500 and remaining only 0.0147727 above the
  run best. Exact metrics `fc9dfc4d…` and the 692,528,817-byte checkpoint `e792bb50…` matched
  remote/mounted; native audit `0dc9b5e7…` passed all 114 tensors / 57,688,576 elements finite and
  nonzero. Post-save steps 38,001–38,050 returned exactly to 7,292/7,294MB buffers and 8,541MB RSS.
  Retention was 36k/37k/38k both sides. Balance was `$40.0002825224`; only Alpha was running, total
  burn was `$0.303/hr`, and mounted disk had 73GB free.
  Step 38,500 then passed with validation still tightly clustered near best: 38,500
  finite/consecutive rows cover 630,784,000 tokens (63.0775%); p10/median was
  3,745.4257/3,879.0059 tok/s and all 386 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2529682/0.2654853 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,466–8,544MB. Train/held-out loss was
  3.2997167/3.2809323, a negligible +0.0017577 wobble from checkpoint 38,000 and only 0.0165303
  above the run best. Exact remote/mounted metrics matched at `bfc478d3…`; the guard remained active
  with zero restarts. Balance was `$39.831379613`; only Alpha was running, total burn was
  `$0.303/hr`, and mounted disk had 72GB free.
  Checkpoint 39,000 then passed with a substantial new run best: 39,000 finite/consecutive rows
  cover 638,976,000 tokens (63.8967%); p10/median was 3,746.2924/3,880.1787 tok/s and all 391
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged
  loss/gradient norm 3.2383468/0.2667200 and held ArrayBuffers at 7,292–7,293MB, external at
  7,294MB, and RSS 8,466–8,541MB. Train/held-out loss was 3.2750547/3.1773408, improving 0.1035915
  from step 38,500 and setting a new run best by 0.0870612 over checkpoint 37,000. Exact metrics
  `f34dbdeb…` and the 692,528,817-byte checkpoint `7f78da25…` matched remote/mounted; native audit
  `ec06cd64…` passed all 114 tensors / 57,688,576 elements finite and nonzero. Post-save steps
  39,001–39,050 returned exactly to 7,292/7,294MB buffers and 8,530MB RSS. Retention was
  37k/38k/39k both sides. Balance was `$39.6626626648`; only Alpha was running, total burn was
  `$0.303/hr`, and mounted disk had 71GB free.
  Step 39,500 then passed with five-batch variance explicitly on watch: 39,500 finite/consecutive
  rows cover 647,168,000 tokens (64.7159%); p10/median was 3,747.1650/3,881.3546 tok/s and all 396
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged
  loss/gradient norm 3.2490023/0.2657663 and held ArrayBuffers/external exactly 7,292/7,294MB, with
  RSS 8,466–8,543MB. Train/held-out loss was 3.2509217/3.2830912, +0.1057504 from the unusually
  sharp checkpoint-39,000 best but only +0.0021588 from step 38,500. This remains established
  five-batch variance, so checkpoint 40,000 is the discriminator and no intervention is justified.
  Exact remote/mounted metrics matched at `f7396b0e…`; the guard remained active with zero restarts.
  Balance was `$39.4938992721`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk
  had 71GB free.
  Checkpoint 40,000 then passed, resolving the wobble to another new run best: 40,000
  finite/consecutive rows cover 655,360,000 tokens (65.5351%); p10/median was
  3,747.9657/3,882.3791 tok/s and all 401 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2550314/0.2709435 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,530–8,542MB. Train/held-out loss was
  3.3543744/3.1690485, improving 0.1140427 from step 39,500 and setting a new run best by 0.0082923
  over checkpoint 39,000. Exact metrics `e83589fd…` and the 692,528,817-byte checkpoint
  `e0f176cb…` matched remote/mounted; native audit `5f48bcd8…` passed all 114 tensors / 57,688,576
  elements finite and nonzero. Post-save steps 40,001–40,050 returned exactly to 7,292/7,294MB
  buffers and 8,542MB RSS. Retention was 38k/39k/40k both sides. Balance was `$39.325022885`; only
  Alpha was running, total burn was `$0.303/hr`, and mounted disk had 70GB free.
  Step 40,500 then passed with five-batch variance on watch: 40,500 finite/consecutive rows cover
  663,552,000 tokens (66.3543%); p10/median was 3,748.6685/3,883.2726 tok/s and all 406 allocator
  samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2557079/0.2689958 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,469–8,543MB. Train/held-out loss was 3.1113625/3.2990192, +0.1299707 from the unusually sharp
  checkpoint-40,000 best and +0.0159281 from the prior high step-39,500 window. This remains
  established five-batch variance, so checkpoint 41,000 is the discriminator and no intervention is
  justified. Exact remote/mounted metrics matched at `94f2cd00…`; the guard remained active with
  zero restarts. Balance was `$39.1802612462`; only Alpha was running, total burn was `$0.303/hr`,
  and mounted disk had 70GB free.
  Checkpoint 41,000 then passed while elevated validation persisted but every hard gate stayed green:
  41,000 finite/consecutive rows cover 671,744,000 tokens (67.1735%); p10/median was
  3,749.5583/3,884.2409 tok/s and all 411 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2326558/0.2729599 and held ArrayBuffers at
  7,292–7,293MB, external at 7,294MB, and RSS 8,466–8,542MB. Train/held-out loss was
  3.0748420/3.3072725, +0.0082532 from step 40,500 and +0.1382240 from the sharp checkpoint-40,000
  best. Two elevated windows are explicit, but remain within earlier recovered oscillation and no
  hard stop fired. Exact metrics `7510063a…` and the 692,528,817-byte checkpoint `1e560e77…`
  matched remote/mounted; native audit `a13ffedc…` passed all 114 tensors / 57,688,576 elements
  finite and nonzero. Post-save steps 41,001–41,050 returned exactly to 7,292/7,294MB buffers and
  8,531MB RSS. Retention was 39k/40k/41k both sides. Balance was `$38.9872020943`; only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 70GB free.
  Step 41,500 then passed with a material recovery in held-out validation: 41,500
  finite/consecutive rows cover 679,936,000 tokens (67.9927%); p10/median was
  3,750.2319/3,884.9894 tok/s and all 416 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2260098/0.2746605 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,490–8,553MB. Train/held-out loss was
  3.2313604/3.2499305, improving 0.0573420 from checkpoint 41,000 and remaining 0.0808820 above the
  sharp checkpoint-40,000 best. This resolves the two-window elevated trend while preserving the
  five-batch-variance caveat. Exact remote/mounted metrics matched at `87d235f5…`; the trainer and
  guard remained healthy with zero guard restarts. Balance was `$38.8425750831`; only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 70GB free.
  Checkpoint 42,000 then passed while validation recovery continued and every save/memory gate stayed
  clean: 42,000 finite/consecutive rows cover 688,128,000 tokens (68.8118%); p10/median was
  3,751.0408/3,885.8204 tok/s and all 421 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2156622/0.2770711 and held
  RSS/ArrayBuffers/external exactly 8,553/7,292/7,294MB. Train/held-out loss was
  3.2169719/3.2275744, improving 0.0223560 from step 41,500 and remaining 0.0585259 above the
  checkpoint-40,000 best. Exact metrics `c4814cca…` and the 692,528,817-byte checkpoint
  `b5354669…` matched remote/mounted; native audit `66c47c81…` passed all 114 tensors / 57,688,576
  elements finite and nonzero. Post-save steps 42,001–42,050 returned exactly to 7,292/7,294MB
  buffers and 8,553MB RSS. Retention was 40k/41k/42k both sides. Balance was `$38.6496333868`; only
  Alpha was running, total burn was `$0.303/hr`, and mounted disk had 70GB free.
  Step 42,500 then passed with modest five-batch variance on watch: 42,500 finite/consecutive rows
  cover 696,320,000 tokens (69.6310%); p10/median was 3,751.6747/3,886.5458 tok/s and all 426
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged
  loss/gradient norm 3.2126758/0.2811326 and held ArrayBuffers/external exactly 7,292/7,294MB, with
  RSS 8,468–8,553MB. Train/held-out loss was 3.2306371/3.2531915, +0.0256171 from checkpoint 42,000
  and +0.0841430 from the sharp checkpoint-40,000 best. This remains established five-batch
  variance, so checkpoint 43,000 is the discriminator. Exact remote/mounted metrics matched at
  `e77ea69d…`; the trainer and guard remained healthy with zero restarts. Balance was
  `$38.5049024867`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk had 70GB
  free.
  Checkpoint 43,000 then passed and recovered the validation wobble: 43,000 finite/consecutive rows
  cover 704,512,000 tokens (70.4502%); p10/median was 3,752.0443/3,886.8866 tok/s and all 431
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged
  loss/gradient norm 3.2033633/0.2803718 and held ArrayBuffers/external exactly 7,292/7,294MB, with
  RSS 8,468–8,542MB. Train/held-out loss was 3.2085137/3.2101865, improving 0.0430050 from step
  42,500 and remaining only 0.0411380 above the checkpoint-40,000 best. Exact metrics `1f08118e…`
  and the 692,528,817-byte checkpoint `3da69bcb…` matched remote/mounted; native audit `93a28428…`
  passed all 114 tensors / 57,688,576 elements finite and nonzero. Post-save steps 43,001–43,050
  returned exactly to 7,292/7,294MB buffers and 8,539–8,540MB RSS. Retention was 41k/42k/43k both
  sides. Balance was `$38.3119838902`; only Alpha was running, total burn was `$0.303/hr`, and
  mounted disk had 69GB free.
  Step 43,500 then passed with continued validation improvement: 43,500 finite/consecutive rows
  cover 712,704,000 tokens (71.2694%); p10/median was 3,752.3466/3,886.9284 tok/s and all 436
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged
  loss/gradient norm 3.1912802/0.2818282 and held ArrayBuffers/external exactly 7,292/7,294MB, with
  RSS 8,471–8,544MB. Train/held-out loss was 3.0849869/3.2006689, improving 0.0095176 from
  checkpoint 43,000 and remaining only 0.0316204 above the checkpoint-40,000 best. Exact
  remote/mounted metrics matched at `66ed2336…`; the trainer and guard remained healthy with zero
  restarts. Balance was `$38.167213268`; only Alpha was running, total burn was `$0.303/hr`, and
  mounted disk had 69GB free.
  Checkpoint 44,000 then passed while one five-batch validation wobble went on watch: 44,000
  finite/consecutive rows cover 720,896,000 tokens (72.0886%); p10/median was
  3,752.5666/3,886.8866 tok/s and all 441 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2118540/0.2841877 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,468–8,543MB. Train/held-out loss was
  3.1902018/3.2532575, +0.0525887 from step 43,500 and +0.0842090 from the checkpoint-40,000 best.
  This is one five-batch wobble after two improving windows; step 44,500 is the discriminator. Exact
  metrics `e1db3751…` and the 692,528,817-byte checkpoint `a64189e6…` matched remote/mounted;
  native audit `c9d89867…` passed all 114 tensors / 57,688,576 elements finite and nonzero.
  Post-save steps 44,001–44,050 returned exactly to 7,292/7,294MB buffers and 8,543MB RSS. Retention
  was 42k/43k/44k both sides. Balance was `$37.9742922272`; only Alpha was running, total burn was
  `$0.303/hr`, and mounted disk had 69GB free.
  Step 46,500 then passed with a material recovery from the post-best validation wobble: 46,500
  finite/consecutive rows cover 761,856,000 tokens (76.1845%); p10/median was
  3,750.0489/3,883.8174 tok/s and all 466 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2058733/0.2950835 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,472–8,545MB. Train/held-out loss was
  3.1847329/3.1806306, improving 0.0463842 from checkpoint 46,000 and remaining only 0.0476633 above
  the sharp step-45,500 best. Exact remote/mounted metrics matched at `5ddd24a5…`; the trainer and
  guard remained healthy with zero restarts. Balance was `$37.1299918752`; only Alpha was running,
  total burn was `$0.303/hr`, and mounted disk had 69GB free.
  Checkpoint 47,000 then passed with continuing validation recovery and a clean save/memory gate:
  47,000 finite/consecutive rows cover 770,048,000 tokens (77.0037%); p10/median was
  3,749.6407/3,883.1745 tok/s and all 471 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1795377/0.2960260 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS exactly 8,545MB. Train/held-out loss was
  3.1541908/3.1714686, improving 0.0091619 from step 46,500 and remaining 0.0385014 above the sharp
  step-45,500 best. Exact remote/mounted metrics matched at `b486260b…`; the 692,528,817-byte
  checkpoint matched at `5c1219a5…`, and native audit `e5243324…` passed all 114 tensors /
  57,688,576 elements finite/nonzero. Post-save rows returned exactly to 7,292/7,294MB buffers and
  8,545MB RSS; retention was 45k/46k/47k both sides. Balance was `$36.9370448011`; only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 69GB free.
  Step 47,500 then passed with a small one-window validation variance and every hard gate clean:
  47,500 finite/consecutive rows cover 778,240,000 tokens (77.8229%); p10/median was
  3,749.0308/3,882.5621 tok/s and all 476 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1798035/0.3013766 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,472–8,555MB. Train/held-out loss was
  3.1647832/3.1811959, +0.0097273 from checkpoint 47,000 and only +0.0482286 above the sharp
  step-45,500 best. Exact remote/mounted metrics matched at `af2c0a38…`; the trainer and guard
  remained healthy with zero restarts. Balance was `$36.7923223344`; only Alpha was running, total
  burn was `$0.303/hr`, and mounted disk had 69GB free.
  Checkpoint 48,000 then passed every integrity gate while a sharp one-window validation spike was
  placed explicitly on watch: 48,000 finite/consecutive rows cover 786,432,000 tokens (78.6421%);
  p10/median was 3,748.7797/3,881.9792 tok/s and all 481 allocator samples reported exactly 34
  slabs/zero overflow. The last 500 rows averaged loss/gradient norm 3.1739363/0.3014846 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,471–8,544MB. Train/held-out loss was
  3.1994004/3.3201346, +0.1389387 from step 47,500 and +0.1871673 above the sharp step-45,500 best.
  This is a serious one-window quality wobble, but train loss, gradients, weights, allocator, and
  memory remain clean; step 48,500 is the discriminator. Exact remote/mounted metrics matched at
  `356609b5…`; the 692,528,817-byte checkpoint matched at `bf298cd4…`, and native audit
  `9cf6692b…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Post-save rows returned
  exactly to 7,292/7,294MB buffers and 8,544MB RSS; retention was 46k/47k/48k both sides. Balance
  was `$36.599386077`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk had 69GB
  free.
  Step 48,500 then materially resolved the checkpoint-48,000 spike: 48,500 finite/consecutive rows
  cover 794,624,000 tokens (79.4613%); p10/median was 3,748.1689/3,881.2362 tok/s and all 486
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged
  loss/gradient norm 3.1566103/0.3011098 and held ArrayBuffers/external exactly 7,292/7,294MB, with
  RSS 8,491–8,555MB. Train/held-out loss was 3.3287439/3.2004976; held-out recovered 0.1196370
  from checkpoint 48,000, sits only +0.0193017 above step 47,500, and +0.0675303 above the sharp
  step-45,500 best. Exact remote/mounted metrics matched at `d9170d63…`; the trainer and guard
  remained healthy with zero restarts. Balance was `$36.4305807622`; only Alpha was running, total
  burn was `$0.303/hr`, and mounted disk had 68GB free.
  Checkpoint 49,000 then passed with moderate validation variance still on aligned watch: 49,000
  finite/consecutive rows cover 802,816,000 tokens (80.2805%); p10/median was
  3,747.2236/3,880.4696 tok/s and all 491 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1829064/0.3071070 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,470–8,555MB. Train/held-out loss was
  3.4289570/3.2353356, +0.0348381 from step 48,500 and +0.1023684 above the sharp step-45,500
  best. Exact remote/mounted metrics matched at `edfdc19b…`; the 692,528,817-byte checkpoint
  matched at `ce31be53…`, and native audit `5d3b64c6…` passed all 114 tensors / 57,688,576 elements
  finite/nonzero. Post-save rows returned exactly to 7,292/7,294MB buffers and 8,541MB RSS;
  retention was 47k/48k/49k both sides. Balance was `$36.2375836381`; only Alpha was running, total
  burn was `$0.303/hr`, and mounted disk had 68GB free.
  Step 49,500 then passed every hard gate while establishing a renewed two-window elevated
  validation phase: 49,500 finite/consecutive rows cover 811,008,000 tokens (81.0997%); p10/median
  was 3,746.8122/3,880.0823 tok/s and all 496 allocator samples reported exactly 34 slabs/zero
  overflow. The last 500 rows averaged loss/gradient norm 3.1433593/0.3170738 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,488–8,552MB. Train/held-out loss was
  3.2038922/3.2817139; held-out is +0.0463783 from checkpoint 49,000, +0.0812163 from step 48,500,
  and +0.1487466 above the sharp step-45,500 best, while remaining below the transient
  checkpoint-48,000 spike. Exact remote/mounted metrics matched at `d212e295…`; the trainer and
  guard remained healthy with zero restarts. Balance was `$36.0929247881`; only Alpha was running,
  total burn was `$0.303/hr`, and mounted disk had 68GB free. Checkpoint 50,000 is the discriminator.
  Checkpoint 50,000 then cleanly resolved the renewed elevated phase: 50,000 finite/consecutive rows
  cover 819,200,000 tokens (81.9189%); p10/median was 3,746.7980/3,879.7418 tok/s and all 501
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged
  loss/gradient norm 3.1791822/0.3079951 and held ArrayBuffers/external exactly 7,292/7,294MB,
  with RSS exactly 8,552MB. Train/held-out loss was 3.1976542/3.2035347; held-out recovered
  0.0781792 from step 49,500, sits only +0.0030372 above step 48,500, and +0.0705675 above the
  sharp step-45,500 best. Exact remote/mounted metrics matched at `e5238d36…`; the 692,528,817-byte
  checkpoint matched at `4bdefac1…`, and native audit `9d6bf76b…` passed all 114 tensors /
  57,688,576 elements finite/nonzero. Post-save rows returned exactly to 7,292/7,294MB buffers and
  8,553MB RSS; retention was 48k/49k/50k both sides. Balance was `$35.8998268919`; only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 68GB free.
  Step 50,500 then passed with post-recovery validation stabilized: 50,500 finite/consecutive rows
  cover 827,392,000 tokens (82.7381%); p10/median was 3,746.7541/3,879.4267 tok/s and all 506
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged
  loss/gradient norm 3.1534839/0.3119011 and held ArrayBuffers/external exactly 7,292/7,294MB,
  with RSS 8,473–8,553MB. Train/held-out loss was 3.1614351/3.2016599; held-out improved 0.0018748
  from checkpoint 50,000, sits only +0.0011623 above step 48,500, and +0.0686926 above the sharp
  step-45,500 best. Exact remote/mounted metrics matched at `1327de1f…`; the trainer and guard
  remained healthy with zero restarts. Balance was `$35.7310038549`; only Alpha was running, total
  burn was `$0.303/hr`, and mounted disk had 68GB free.
  Checkpoint 51,000 then passed with a moderate one-window validation wobble on aligned watch:
  51,000 finite/consecutive rows cover 835,584,000 tokens (83.5572%); p10/median was
  3,747.0509/3,879.4109 tok/s and all 511 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1456441/0.3094241 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,470–8,545MB. Train/held-out loss was
  3.2150030/3.2513193, +0.0496593 from step 50,500 and +0.1183520 above the sharp step-45,500
  best. Exact remote/mounted metrics matched at `57475440…`; the 692,528,817-byte checkpoint
  matched at `e5aeb795…`, and native audit `3047e2b1…` passed all 114 tensors / 57,688,576 elements
  finite/nonzero. Post-save rows returned exactly to 7,292/7,294MB buffers and 8,542MB RSS;
  retention was 49k/50k/51k both sides. Balance was `$35.5621934067`; only Alpha was running, total
  burn was `$0.303/hr`, and mounted disk had 67GB free.
  Step 51,500 then recovered validation to near-run-best: 51,500 finite/consecutive rows cover
  843,776,000 tokens (84.3764%); p10/median was 3,747.4460/3,879.6410 tok/s and all 516 allocator
  samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1485280/0.3142221 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,471–8,546MB. Train/held-out loss was 3.1850824/3.1555889; held-out improved 0.0957304 from
  checkpoint 51,000 and is only +0.0226216 above the sharp step-45,500 best. Exact remote/mounted
  metrics matched at `e841965c…`; the trainer and guard remained healthy with zero restarts.
  Balance was `$35.3933533196`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk
  had 67GB free.
  Checkpoint 52,000 then set a new run best: 52,000 finite/consecutive rows cover 851,968,000 tokens
  (85.1956%); p10/median was 3,747.7495/3,879.6506 tok/s and all 521 allocator samples reported
  exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1574955/0.3157661 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,527–8,545MB. Train/held-out loss was 3.0159855/3.1257953; held-out improved 0.0297936 from step
  51,500 and set a new run best by 0.0071720 over step 45,500. Exact remote/mounted metrics matched
  at `3bc0a40c…`; the 692,528,817-byte checkpoint matched at `9a2c585d…`, and native audit
  `e7fd7fa9…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Post-save rows returned
  exactly to 7,292/7,294MB buffers and 8,545MB RSS; retention was 50k/51k/52k both sides. Balance
  was `$35.2244717991`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk had
  66GB free.
  Step 52,500 then set another run best: 52,500 finite/consecutive rows cover 860,160,000 tokens
  (86.0148%); p10/median was 3,748.1008/3,879.6693 tok/s and all 526 allocator samples reported
  exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1478990/0.3178515 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,470–8,547MB. Train/held-out loss was 3.1507206/3.1009022; held-out improved 0.0248931 from
  checkpoint 52,000 and became the new run best. Exact remote/mounted metrics matched at
  `8972c1b3…`; the trainer and guard remained healthy with zero restarts. Balance was
  `$35.0556843897`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk had 66GB
  free.
  Checkpoint 53,000 then passed with a moderate one-window rebound from the new best: 53,000
  finite/consecutive rows cover 868,352,000 tokens (86.8340%); p10/median was
  3,748.3551/3,879.7285 tok/s and all 531 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1392291/0.3207649 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,534–8,546MB. Train/held-out loss was
  3.1202853/3.1644977, +0.0635955 from the exceptional step-52,500 best. Exact remote/mounted
  metrics matched at `562e8403…`; the 692,528,817-byte checkpoint matched at `b2cb6865…`, and
  native audit `c56206c9…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Post-save
  rows returned exactly to 7,292/7,294MB buffers and held RSS at 8,546MB; retention was
  51k/52k/53k both sides. Balance was `$34.8867962693`; only Alpha was running, total burn was
  `$0.303/hr`, and mounted disk had 66GB free.
  Step 53,500 then passed with a two-window elevated validation phase on watch: 53,500
  finite/consecutive rows cover 876,544,000 tokens (87.6532%); p10/median was
  3,748.5257/3,879.8256 tok/s and all 536 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1241615/0.3207876 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,474–8,548MB. Train/held-out loss was
  3.2312453/3.2067587; held-out is +0.0422610 from checkpoint 53,000 and +0.1058565 from the sharp
  step-52,500 best, while remaining 0.1133759 below the earlier checkpoint-48,000 spike. Exact
  remote/mounted metrics matched at `41f392fa…`; the trainer and guard remained healthy with zero
  restarts. Balance was `$34.7179635157`; only Alpha was running, total burn was `$0.303/hr`, and
  mounted disk had 65GB free.
  Checkpoint 54,000 then passed while the elevated validation phase eased but remained on watch:
  54,000 finite/consecutive rows cover 884,736,000 tokens (88.4724%); p10/median was
  3,748.7694/3,879.6273 tok/s and all 541 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1495170/0.3400912 and held
  ArrayBuffers/external exactly 7,292/7,294MB with RSS exactly 8,546MB. Train/held-out loss was
  3.2111425/3.1928943; held-out improved 0.0138644 from step 53,500 but remained +0.0919921 above
  the sharp step-52,500 best. Exact remote/mounted metrics matched at `34a0ab36…`; the
  692,528,817-byte checkpoint matched at `3fb1913a…`, and native audit `6fc1ea2b…` passed all 114
  tensors / 57,688,576 elements finite/nonzero. Post-save rows returned exactly to
  7,292/7,294MB buffers and held RSS at 8,546MB; retention was 52k/53k/54k both sides. Balance was
  `$34.5250273194`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk had 65GB
  free.
  Step 54,500 then passed while the elevated validation plateau eased marginally and remained on
  watch: 54,500 finite/consecutive rows cover 892,928,000 tokens (89.2916%); p10/median was
  3,748.7665/3,879.5157 tok/s and all 546 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1415791/0.3262672; ArrayBuffers/external varied
  only 1MB at 7,292–7,293/7,294–7,295MB and RSS stayed within 8,471–8,546MB. Train/held-out loss
  was 3.0450501/3.1912791; held-out improved 0.0016152 from checkpoint 54,000 and 0.0154796 from
  step 53,500, while remaining +0.0903769 above the sharp step-52,500 best. Exact remote/mounted
  metrics matched at `500ec2ef…`; the trainer and guard remained healthy with zero restarts.
  Balance was `$34.3803019805`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk
  had 65GB free.
  Checkpoint 55,000 then materially resolved the elevated validation plateau toward baseline: 55,000
  finite/consecutive rows cover 901,120,000 tokens (90.1108%); p10/median was
  3,748.6685/3,879.1190 tok/s and all 551 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1226775/0.3276708, held
  ArrayBuffers/external exactly 7,292/7,294MB, and held RSS within 8,540–8,547MB. Train/held-out
  loss was 3.1819386/3.1667840; held-out improved 0.0244951 from step 54,500 and was only
  +0.0022863 above checkpoint 53,000, while remaining +0.0658818 above the sharp step-52,500 best.
  Exact remote/mounted metrics matched at `092e479f…`; the 692,528,817-byte checkpoint matched at
  `95e8cd31…`, and native audit `26888e73…` passed all 114 tensors / 57,688,576 elements
  finite/nonzero. Post-save rows returned exactly to 7,292/7,294MB buffers and held RSS at 8,547MB;
  retention was 53k/54k/55k both sides. Balance was `$34.1873456174`; only Alpha was running, total
  burn was `$0.303/hr`, and mounted disk had 64GB free.
  Step 55,500 then resolved the elevated validation plateau: 55,500 finite/consecutive rows cover
  909,312,000 tokens (90.9299%); p10/median was 3,748.7797/3,878.9848 tok/s and all 556 allocator
  samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1314820/0.3289836 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,471–8,548MB. Train/held-out loss was 3.0617723/3.1577666; held-out improved 0.0090174 from
  checkpoint 55,000, 0.0335125 from step 54,500, and was 0.0067311 better than checkpoint 53,000,
  while remaining +0.0568644 above the sharp step-52,500 best. Exact remote/mounted metrics matched
  at `f8a382d0…`; the trainer and guard remained healthy with zero restarts. Balance was
  `$34.0426110506`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk had 64GB
  free.
  Checkpoint 56,000 then continued the validation recovery near run best: 56,000 finite/consecutive
  rows cover 917,504,000 tokens (91.7491%); p10/median was 3,749.1632/3,879.2154 tok/s and all 561
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged
  loss/gradient norm 3.1271782/0.3319130 and held ArrayBuffers/external exactly 7,292/7,294MB,
  with RSS 8,471–8,547MB. Train/held-out loss was 3.2305961/3.1364248; held-out improved 0.0213418
  from step 55,500 and 0.0303592 from checkpoint 55,000, leaving only +0.0355226 above the sharp
  step-52,500 best. Exact remote/mounted metrics matched at `15b11de0…`; the 692,528,817-byte
  checkpoint matched at `41923a11…`, and native audit `b4b174ba…` passed all 114 tensors /
  57,688,576 elements finite/nonzero. Post-save rows returned exactly to 7,292/7,294MB buffers and
  held RSS at 8,547MB; retention was 54k/55k/56k both sides. Balance was `$33.8496793765`; only
  Alpha was running, total burn was `$0.303/hr`, and mounted disk had 64GB free.
  Step 56,500 then returned held-out loss to within 0.008 of the run best: 56,500 finite/consecutive
  rows cover 925,696,000 tokens (92.5683%); p10/median was 3,749.6664/3,879.5791 tok/s and all 566
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.1224420/0.3326293 and held RSS/ArrayBuffers/external exactly at
  8,547/7,292/7,294MB. Train/held-out loss was 3.2060032/3.1087756; held-out improved 0.0276492
  from checkpoint 56,000 and is only +0.0078734 above the sharp step-52,500 run best. Exact
  remote/mounted metrics matched at `adfd1a15…`. Balance was `$33.7049310597`; only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 62GB free. The recovery pod's exact SFT
  and frozen-eval inputs were also fully staged and hash-verified, leaving 7.6GB free on `/runpod`.
  Checkpoint 57,000 then set a new run best: 57,000 finite/consecutive rows cover 933,888,000 tokens
  (93.3875%); p10/median was 3,750.2319/3,879.8896 tok/s and all 571 allocator samples reported
  exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1198608/0.3377949 and held RSS/ArrayBuffers/external exactly at 8,547/7,292/7,294MB.
  Train/held-out loss was 3.0678167/3.0660259; held-out improved 0.0427497 from step 56,500 and set
  a new run best by 0.0348763 over step 52,500. Exact remote/mounted metrics matched at
  `f3d0c063…`; the 692,528,817-byte checkpoint matched at `eae1679e…`, and native audit
  `d6a277a5…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Post-save rows held the
  exact same memory baseline; retention was 55k/56k/57k both sides. Balance was `$33.5119787299`;
  only Alpha was running, total burn was `$0.303/hr`, and mounted disk had 63GB free.
  Step 57,500 then passed with a moderate one-window rebound after the new best: 57,500
  finite/consecutive rows cover 942,080,000 tokens (94.2067%); p10/median was
  3,750.7619/3,880.2897 tok/s and all 576 allocator samples reported exactly 34 slabs/zero
  overflow. The last 500 rows averaged loss/gradient norm 3.0976251/0.3547236;
  ArrayBuffers/external held exactly at 7,292/7,294MB and RSS stayed within 8,474–8,547MB.
  Train/held-out loss was 3.2421117/3.1012069, +0.0351810 from the new checkpoint-57,000 best and
  only +0.0003047 above the former step-52,500 best. Exact remote/mounted metrics matched at
  `b5120435…`. Balance was `$33.3672945799`; only Alpha was running, total burn was `$0.303/hr`,
  and mounted disk had 62GB free.
  Checkpoint 58,000 then passed while a two-window elevated validation phase returned to watch:
  58,000 finite/consecutive rows cover 950,272,000 tokens (95.0259%); p10/median was
  3,751.1933/3,880.8205 tok/s and all 581 allocator samples reported exactly 34 slabs/zero
  overflow. The last 500 rows averaged loss/gradient norm 3.0931357/0.3387721;
  ArrayBuffers/external held exactly at 7,292/7,294MB and RSS stayed within 8,535–8,546MB.
  Train/held-out loss was 3.1479688/3.1600277, +0.0588208 from step 57,500 and +0.0940019 from
  the checkpoint-57,000 best. Exact remote/mounted metrics matched at `5f6bdb84…`; the
  692,528,817-byte checkpoint matched at `85b949ff…`, and native audit `97580626…` passed all 114
  tensors / 57,688,576 elements finite/nonzero. Post-save rows returned exactly to the memory
  baseline; retention was 56k/57k/58k both sides. Balance was `$33.1743314336`; only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 62GB free.
  Step 58,500 then resolved the elevated validation phase: 58,500 finite/consecutive rows cover
  958,464,000 tokens (95.8451%); p10/median was 3,751.6908/3,881.2779 tok/s and all 586 allocator
  samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1321392/0.3434877; ArrayBuffers/external held exactly at 7,292/7,294MB and RSS stayed within
  8,493–8,556MB. Train/held-out loss was 3.2500243/3.1015048; held-out improved 0.0585229 from
  checkpoint 58,000, is only +0.0006026 above the former step-52,500 best, and remains +0.0354789
  above the checkpoint-57,000 run best. Exact remote/mounted metrics matched at `6dbd87f6…`.
  Balance was `$33.0295856225`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk
  had 62GB free.
  Checkpoint 59,000 then continued the validation recovery: 59,000 finite/consecutive rows cover
  966,656,000 tokens (96.6643%); p10/median was 3,752.0499/3,881.7374 tok/s and all 591 allocator
  samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1191765/0.3458788 and held RSS/ArrayBuffers/external exactly at 8,556/7,292/7,294MB.
  Train/held-out loss was 2.8956242/3.0951898; held-out improved 0.0063150 from step 58,500 and is
  only +0.0291639 above the checkpoint-57,000 run best. Exact remote/mounted metrics matched at
  `17af1cc2…`; the 692,528,817-byte checkpoint matched at `a96b22e2…`, and native audit
  `4104dcc1…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Post-save rows held the
  exact memory baseline; retention was 57k/58k/59k both sides. Balance was `$32.8366250428`; only
  Alpha was running, total burn was `$0.303/hr`, and mounted disk had 62GB free.
  Step 59,500 then returned validation near the run best: 59,500 finite/consecutive rows cover
  974,848,000 tokens (97.4835%); p10/median was 3,752.3330/3,882.0040 tok/s and all 596 allocator
  samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.0926898/0.3416984; ArrayBuffers/external held exactly at 7,292/7,294MB and RSS stayed within
  8,485–8,557MB. Train/held-out loss was 2.9711595/3.0791674; held-out improved 0.0160223 from
  checkpoint 59,000 and is only +0.0131415 above the checkpoint-57,000 run best. Exact
  remote/mounted metrics matched at `3cccaf99…`. Balance was `$32.6918561928`; only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 62GB free.
  Checkpoint 60,000 then recorded a late validation wobble with every hard gate still clean: 60,000
  finite/consecutive rows cover 983,040,000 tokens (98.3026%); p10/median was
  3,752.6690/3,882.1724 tok/s and all 601 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1206065/0.3466359; RSS stayed within
  8,527–8,556MB while ArrayBuffers/external held exactly at 7,292/7,294MB. Train/held-out loss was
  3.1976528/3.2482990, +0.1691316 from step 59,500 and +0.1822731 above the checkpoint-57,000 run
  best. Exact metrics `d15f007c…` and the 692,528,817-byte checkpoint `cd124a9c…` matched
  remote/mounted; native audit `b66644dc…` passed all 114 tensors / 57,688,576 elements
  finite/nonzero. Post-save rows 60,001–60,050 held the reclaimed 8,534–8,535/7,292/7,294MB
  RSS/ArrayBuffers/external baseline. Retention was 58k/59k/60k both sides. Balance was
  `$32.4748066537`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk had 62GB
  free.
  Step 60,500 then decisively resolved the late wobble and set a new run best: 60,500
  finite/consecutive rows cover 991,232,000 tokens (99.1218%); p10/median was
  3,752.8326/3,882.1851 tok/s and all 606 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1226713/0.3479907; ArrayBuffers/external held
  exactly at 7,292/7,294MB and RSS stayed within 8,475–8,549MB. Train/held-out loss was
  3.1089549/3.0491906; held-out recovered 0.1991084 from checkpoint 60,000 and set a new run best by
  0.0168353 over checkpoint 57,000. The final batch's pre-clip grad norm 1.278 was correctly clipped
  to coefficient 0.782; the aligned mean and every finite/system gate remained healthy. Exact
  remote/mounted metrics matched at `c9179f8f…`. Balance was `$32.3541754074`; only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 62GB free.
  Checkpoint 61,000 then passed: 61,000 finite/consecutive rows cover 999,424,000 tokens (99.9410%);
  p10/median was 3,752.9627/3,882.1910 tok/s and all 611 allocator samples reported exactly 34
  slabs/zero overflow. The last 500 rows averaged loss/gradient norm 3.1076607/0.3496741 and held
  RSS/ArrayBuffers/external exactly at 8,547/7,292/7,294MB. Train/held-out loss was
  3.2091236/3.1908423, a rebound from the exceptional step-60,500 best but still 0.0574567 better
  than checkpoint 60,000. Exact metrics `cf2a2e4c…` and checkpoint `8b2872ab…` matched
  remote/mounted; native audit `c1171427…` passed all 57,688,576 parameters finite/nonzero.
  The `e561f66` flagship then completed all 61,036 rows and exactly 1,000,013,824 tokens. Terminal
  analyzer `5d65e518…` passed exact selector/manifest/tokenizer/contract binding, p10/median
  3,753.1721/3,882.3479 tok/s after warmup, 612 complete allocator samples with zero overflow,
  final/last-100 train loss 2.9974854/3.1011362, final-three validation mean 3.1367731, and all
  57,688,576 terminal parameters finite/nonzero. Terminal checkpoint is 692,528,817 bytes at
  `08e14fa9…`; canonical metrics are `7ff9feec…`.
  Terminal analysis exposed a real contract bug: the `e561f66` trainer evaluated cadence multiples
  only while the analyzer correctly required off-cadence terminal step 61,036. `4c5d1aa` fixes all
  future terminal cadence. A 36-step replay exercised the fix but produced a different checkpoint
  (`039a260d…`) because Vulkan reductions are not bit-deterministic, so it was rejected as canonical
  and preserved as named evidence. `c333bf2` adds a fail-closed eval-only repair: it loaded the sealed
  original `08e14fa9…` checkpoint, ran exactly five validation batches and zero training steps,
  measured terminal val loss 3.1702865, and changed only `valLoss` on row 61,036. Repair evidence is
  `56e77083…`; original metrics remain preserved at `c383d24b…`.
  Step 44,500 then passed while elevated validation persisted but every hard gate remained green:
  44,500 finite/consecutive rows cover 729,088,000 tokens (72.9078%); p10/median was
  3,752.2116/3,886.4798 tok/s and all 446 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2248817/0.2865226 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,469–8,545MB. Train/held-out loss was
  3.2194901/3.2662046, +0.0129471 from checkpoint 44,000 and +0.0971561 from the checkpoint-40,000
  best. Two elevated windows are explicit, but this pattern has recovered before and no hard stop
  fired; checkpoint 45,000 is the discriminator. Exact remote/mounted metrics matched at
  `56436775…`; the trainer and guard remained healthy with zero restarts. Balance was
  `$37.8295331549`; only Alpha was running, total burn was `$0.303/hr`, and mounted disk had 69GB
  free.
  Checkpoint 45,000 then passed while elevated validation eased but remained on watch: 45,000
  finite/consecutive rows cover 737,280,000 tokens (73.7270%); p10/median was
  3,751.7456/3,885.8743 tok/s and all 451 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2021399/0.2907239 and held
  RSS/ArrayBuffers/external at 8,543–8,544/7,292/7,294MB. Train/held-out loss was
  3.2578907/3.2536933, improving 0.0125113 from step 44,500 but remaining 0.0846448 above the
  checkpoint-40,000 best. Exact metrics `9e57f4e1…` and the 692,528,817-byte checkpoint
  `dd8852f0…` matched remote/mounted; native audit `372487d9…` passed all 114 tensors / 57,688,576
  elements finite and nonzero. Post-save steps 45,001–45,050 returned exactly to 7,292/7,294MB
  buffers and 8,544MB RSS. Retention was 43k/44k/45k both sides. Balance was `$37.6365491698`; only
  Alpha was running, total burn was `$0.303/hr`, and mounted disk had 69GB free.
  Step 45,500 then passed and resolved the elevated phase to a substantial new run best: 45,500
  finite/consecutive rows cover 745,472,000 tokens (74.5462%); p10/median was
  3,751.4171/3,885.4225 tok/s and all 456 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.1922874/0.2969672 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,472–8,545MB. Train/held-out loss was
  3.1899173/3.1329673, improving 0.1207261 from checkpoint 45,000 and setting a new run best by
  0.0360812 over checkpoint 40,000. Exact remote/mounted metrics matched at `9bd00c17…`; the trainer
  and guard remained healthy with zero restarts. Balance was `$37.4677396994`; only Alpha was
  running, total burn was `$0.303/hr`, and mounted disk had 69GB free.
  Checkpoint 46,000 then passed with one-window variance from the unusually sharp new best while all
  hard gates stayed clean: 46,000 finite/consecutive rows cover 753,664,000 tokens (75.3654%);
  p10/median was 3,750.7619/3,884.4729 tok/s and all 461 allocator samples reported exactly 34
  slabs/zero overflow. The last 500 rows averaged loss/gradient norm 3.2024244/0.2999512 and held
  RSS/ArrayBuffers/external at 8,543–8,544/7,292/7,294MB. Train/held-out loss was
  3.2123423/3.2270148, +0.0940475 from step 45,500 but still 0.0266785 better than checkpoint 45,000.
  Exact metrics `b42b3010…` and the 692,528,817-byte checkpoint `1ba70b29…` matched remote/mounted;
  native audit `d2d2f123…` passed all 114 tensors / 57,688,576 elements finite and nonzero.
  Post-save steps 46,001–46,050 returned exactly to 7,292/7,294MB buffers and 8,544MB RSS. Retention
  was 44k/45k/46k both sides. Balance was `$37.2747729474`; only Alpha was running, total burn was
  `$0.303/hr`, and mounted disk had 69GB free.
  Checkpoint 31,000 then passed with a third consecutive held-out run best and crossed halfway:
  31,000 finite/consecutive rows cover 507,904,000 tokens (50.7897%); p10/median was
  3,732.8800/3,862.6160 tok/s and all 311 allocator samples reported exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.3137590/0.2355993 and held
  RSS/ArrayBuffers/external exactly 8,490/7,292/7,294MB. Train/held-out loss was
  3.3144221/3.3412647, improving 0.0183384 from step 30,500. Exact metrics `f6092046…` and the
  692,528,817-byte checkpoint `8372b814…` match remote/mounted; native audit `948de04f…` passed all
  114 tensors / 57,688,576 elements finite and nonzero. The save released all 228 clones; steps
  31,001–31,050 returned to exactly 7,292/7,294MB ArrayBuffers/external while RSS settled only 48MB
  higher at 8,538MB. Safe retention is exactly 29k/30k/31k on both sides. Balance was
  `$43.5891647432`; only Alpha was running and total account burn returned to `$0.303/hr`.
  Step 31,500 then passed with a substantial new held-out run best: 31,500 finite/consecutive rows
  cover 516,096,000 tokens (51.6089%); p10/median was 3,733.7679/3,863.6001 tok/s and all 316
  allocator samples reported exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.3260559/0.2355921 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,467–8,538MB. Train/held-out loss was 3.4092531/3.2892870, improving 0.0519777 from checkpoint
  31,000. Exact remote/mounted metrics matched at `c63ee5b8…`; guard remained active/zero-restart.
  Balance was `$43.3246554467`, only Alpha was running, and total account burn remained `$0.303/hr`.
  Checkpoint 32,000 then passed: 32,000 finite/consecutive rows cover 524,288,000 tokens (52.4281%);
  p10/median was 3,734.6678/3,864.4482 tok/s and all 321 allocator samples reported exactly 34
  slabs/zero overflow. The last 500 rows averaged loss/gradient norm 3.3093260/0.2391542 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,537–8,538MB. Train/held-out loss was
  3.3595934/3.3350753, +0.0457882 from the unusually strong 31,500 window but 0.0061894 better than
  checkpoint 31,000. Exact metrics `08eb6938…` and the 692,528,817-byte checkpoint `e82ac311…`
  match remote/mounted; native audit `79ba1fa4…` passed all 114 tensors / 57,688,576 elements finite
  and nonzero. The save returned ArrayBuffers directly to baseline; steps 32,001–32,050 held
  7,292/7,294MB ArrayBuffers/external and RSS only 1MB higher at 8,539MB. Retention is exactly
  30k/31k/32k on both sides. Balance was `$42.9987368539`, total burn `$0.303/hr`, mounted disk 84GB
  free.
  Step 32,500 then passed: 32,500 finite/consecutive rows cover 532,480,000 tokens (53.2473%);
  p10/median was 3,735.5363/3,865.5490 tok/s and all 326 allocator samples reported exactly 34
  slabs/zero overflow. The last 500 rows averaged loss/gradient norm 3.3296892/0.2432705 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,466–8,539MB. Train/held-out loss was
  3.1392970/3.3509093, +0.0158340 from checkpoint 32,000 and +0.0616222 from the sharp step-31,500
  best; all invariants remained green, making this normal five-batch variance pending checkpoint
  33,000. Exact remote/mounted metrics matched at `441c237d…`; guard remained active/zero-restart.
  Balance was `$42.8003199855`; total burn was `$0.75/hr` because unrelated Wajarri pod
  `b21dbqjy0t3gir` was running at `$0.44/hr`. Alpha remained `$0.22/hr`; the unrelated pod was not
  touched.
- **SFT**: assistant-only masked loss on the Stage-4 chat mix, 1-2 epochs, lr swept {1e-4, 3e-4, 1e-3}
  (SmolLM2-360M SFT reference = 1e-3 × 2 epochs cosine), then re-run the FULL frozen eval + base-vs-chat
  regression (does SFT destroy LM quality? report). `--initCheckpoint` (`55c86db`) loads base weights
  with model-compatibility validation while resetting step/RNG/optimizer/schedule; continuation resume
  remains a distinct, mutually exclusive path. Compatibility now fails closed on RMSNorm/LayerNorm,
  RoPE/learned positions, RoPE theta, tying, and soft-cap as well as every dimension (`6b460e4`).
  `run_flagship_sft.sh` (`7636ad2`) contracts the selected sweep LR and exact one-epoch shape: 485,150
  train + 26,278 validation conversations, 30,322 batches / 496,795,648 padded tokens. Its input verifier
  independently hashes and line-counts the corpus, requires the length/mask audits, reconciles the binary
  checkpoint tensor table and byte length, scans all 57,688,576 base parameters for finiteness, and binds
  corpus/audits/tokenizer/base checkpoint/current commit into the resumable run contract.
  `analyze_flagship_sft.ts` closes the other end: it binds the three-way SFT selector, all 30,322 finite
  rows and 61 aligned validations, full allocator cadence/zero overflow, and the existing streaming
  finite/nonzero verifier's audit of the terminal chat checkpoint.
  SFT resume also preserves the original `initCheckpointPath` in the rewritten `config.json` and fails
  closed if that origin provenance is absent, so resumed pilots/full runs remain auditable to the base.
  The choice itself is contracted in `b24c18a`: three 2,000-step / 32,768,000-token pilots over
  `{1e-4,3e-4,1e-3}` share exact inputs and eight aligned validations; `analyze_sft_lr_sweep.ts` chooses
  the lowest final-three held-out mean. The one-epoch launcher now requires and verifies that selector
  report, including its commit and all six input hashes, rather than accepting any allowed number.
  The selected full run's checkpoint-2,000 recovery gate passed at held-out loss 1.7896707, 3,945 tok/s,
  complete zero-overflow allocator telemetry, and exact remote/mounted checkpoint SHA `1878ed9e...`.
  Independent audit `477cde8f...` scanned all 57,688,576 parameters finite/nonzero. Post-save live
  buffers returned to baseline through step 2,100; RSS retained a stable ~314MB allocator-page plateau,
  and checkpoint 3,000 resolved the memory discriminator: pre/post-save live buffers were unchanged and
  RSS rose only 1–4MB, not another checkpoint-sized plateau. Checkpoint 3,000 is exact remote/mounted at
  `5ad80097...`; audit `7af019aa...` passed all parameters; exact metrics prefix `286052e7...` proves
  3,000 finite/consecutive rows, 31 allocator samples/zero overflow, p10/median
  3,801.65/3,927.61 tok/s, and held-out loss 1.7948370. Separate three-prompt non-frozen previews at
  6.6% and 9.9% of the epoch remained mixed/repetitive and did not alter the run or expose frozen data.
- **Ops discipline** (box CLAUDE.md rules apply): verify-it-actually-works — measure real tok/s from
  metrics deltas not logs; watchdog terminates any pod whose checkpoint stream stalls 30 min; every
  run resumable (`--resume`); no fire-and-forget. `99a9116` bounds ~693MB checkpoint growth with matched
  three-copy remote/local retention: remote pruning requires a byte+SHA mirror proof, and each subsequent
  local deletion is itself hash-ledgered before and after unlink. The policy is opt-in and rejects fewer
  than three recovery points. `6d92470` adds a separate fail-closed terminal watcher: live-pod preflight
  and wrong-source/premature-finalization rejection passed, and service
  `alpha2-flagship-sft-finalizer-20260728` proved consecutive 3,250→3,300-row polls with the exact trainer
  PID. After a clean 30,322-step exit it automates terminal audit/analysis, frozen eval, pair analysis,
  HF export/parity, full remote-manifest/local-hash verification, and only then scoped pod removal.
  Machine D3 failure is preserved rather than published; semantic review and chat upload remain manual.
- **Gate G5 = D3 chat bar.** If quality is word-salad at the bar, we do NOT ship a chat model; we ship
  the base model with an honest card and the ledger records what a bigger budget would change.

### Stage 6 — Ship (box only, $0 GPU)
- [x] TS **safetensors writer** (~25 lines, spec is trivial: u64 header-len + JSON + raw LE f32) with
      `__metadata__ {"format":"pt"}`; round-trip verified against Python `safetensors` on the box.
- [x] `config.json` (llama; explicit `num_key_value_heads`, `head_dim`, `rms_norm_eps=1e-5`, flat
      `rope_theta`, `tie_word_embeddings: true` + **omit lm_head from the file** — the #1 silent-garbage
      pitfall), `generation_config.json`, `chat_template.jinja` with `{% generation %}` markers.
- [▶] Model cards: base card is public with the from-scratch story, exact architecture/training/cost,
      ODC-By pretraining provenance, full failed base eval, and limitations. Chat card remains pending
      the terminal frozen evaluation and semantic review; it must include Apache-2.0 / CC-BY-4.0 SFT
      attribution and the true base-vs-chat table.
- [▶] HF publication: `ajaxdavis/alpha-60m-base` is public at Hub commit `8693cb4c...`; anonymous
      empty-cache stock-Transformers CPU cold load passed both plain-text and message-list pipelines,
      exact 57,688,576 parameters, and safetensors SHA `d0aa2ccd...`. Publication proof is sealed under
      `/mnt/donto-data/alpha-runs/hf-base-publication-c333bf2-20260728/`. Chat upload/cold-load and the
      final `alpha2` release tag remain pending G5/D3.
- [ ] Stretch (post-D2): GGUF via `convert_hf_to_gguf.py` (needs the `get_vocab_base_pre` patch for a
      custom vocab — patch to `"gpt-2"` pre since we adopt the GPT-2 split regex); refresh the HF Space
      (apps/hf) to serve the new model with proper chat template + EOS stop; update apps/web `/v1` route
      (currently joins messages with `\n` and never stops on EOS — demo-killer).

## 5. Budget ledger (update on every spend; hard ceiling $70.21)

| Item | Est. | Actual |
|---|---|---|
| Beachhead probes (2026-07-22, 3090 ~1.5h) | $0.50 | ~$0.40 |
| Stage 0-1 smoke + parity runs | $4 | ~$0.30 incremental through G1 (live pod; reconcile at termination) |
| Stage 2 profiling + 6h soak | $8 | PASS; final soak GPU time ≈6.44h / ≈$1.42; account balance $66.736685594 at 20:45 UTC while pod remained live for G3; account burn also includes unrelated stopped volumes |
| Stage 3 pilots (2× 100M-token) | $4 | Llama half launched on certified `c95f81b`; balance $66.5346286162 at 21:24 UTC, account burn $0.301/hr incl. unrelated volumes |
| Stage 5 lr sweeps (3× 100M-token) | $5 | |
| Stage 5 pretrain 1B tok @ ≥3K tok/s | $20-25 | |
| Stage 5 SFT + evals | $3 | |
| Reserve / re-runs | $15 | |

GPU preference order (price × proven-Vulkan odds): **RTX 3090 $0.22 (proven today) → A5000 $0.16 (probe
first) → 4090 $0.34 (if compute-bound) → A40 $0.30-0.35 (48GB headroom)**. Community cloud, on-demand;
spot only with the checkpoint-puller running. NOTE: 4 stopped mobtranslate/migmaq pods are burning
~$1.80/day in volume storage — user decision, not part of this program.

## 6. Standing decisions (do not relitigate without new evidence)

1. **From-scratch invariant**: all training compute through Helios. The recommendation doc's
   "import SmolLM2-360M + HF SFT" track is REJECTED as the mission (violates the soul constraint).
   Weight-import survives only as the optional golden-logit validation idea, superseded by G3's
   export-and-compare test which needs no foreign weights.
2. **Publish as a standard arch** (Llama-form; Qwen3-form fallback if QK-norm proves necessary) rather
   than ALPH-in-a-Space. The Space becomes a consumer of the standard artifact, not the artifact.
3. **f32 end-to-end for the flagship.** fp16/coop-matmul re-enters only after G1-class parity proof at
   scale, as a separately gated experiment. (L4 auto-profile force-enables fp16 — always pass
   `--fp16=false`; kill that default in Stage 1.)
4. **lr truth**: domains.ts values are canonical; `scripts/train-nanochat.sh`'s 6e-4 is documented-divergent
   and gets fixed in Stage 1. Post-bugfix lr is re-swept (G5), not inherited.
5. **Tokenizer**: Alpha-native byte-level BPE (12,288) with GPT-2 split regex + exporter. NOT SmolLM2's
   49K vocab (wastes a tiny model's budget; foreign tokenizer weakens the from-scratch story).
6. **No benchmark contamination**: MMLU/GSM8K-style sets never enter training mixes; evals are frozen
   pre-flagship.
7. **Preference optimization (DPO/GRPO), retrieval, self-play: OUT of scope** until D1-D3 ship.

## 7. Top risks → mitigations

| Risk | Mitigation |
|---|---|
| Vulkan lottery on community hosts (egress AND driver variance) | bootstrap probes vulkaninfo-equivalent before any setup; auto-terminate + redeploy elsewhere (per-second billing makes probes ~free); everything rsynced from box, no pod-side github/apt |
| Slab wiring doesn't yield 3K tok/s | G2 decision point: shrink model / cut tokens — explicitly, in the ledger; program still ships smaller D1/D2 |
| NaN root cause resists diagnosis | parity harness bisects op-by-op; worst case: gelu FFN fallback flagship (gelu is Llama-config-expressible via `hidden_act`) — records as ledger decision |
| 60M @ 1B tokens is still word-salad | honest D3 bar: ship base-only + truthful card; the chat bar consciously targets format-coherent small-child quality, not knowledge |
| Community pod dies mid-run | checkpoint every eval interval + box-side puller + `--resume`; spot only with puller verified |
| Repo secrets already public | Stage 1 scrub + rotation before any publicity |
| JS heap/string limits on big corpora | shard corpus files ≤2GB; loader already chunks; token cache Int32 = 4 bytes/token budgeted |

## 8. Immediate next actions (current 2026-07-28)

1. [x] Certify `c333bf2` on the live RTX 3090 as the SFT source: 46/46 GPU-gated tests passed with
   zero skips/failures/todos. This advances only the SFT stage boundary and preserves `e561f66` as
   the exact base-pretrain source.
2. [x] Complete the contracted SFT LR sweep. All three pilots completed 2,000/2,000 finite rows with
   complete allocator cadence and zero overflow. Strict PASS report `06243d36...` selects `3e-4`:
   final-three means 1.7839965 (`3e-4`), 1.8391179 (`1e-3`), 1.8586174 (`1e-4`).
3. [▶] Complete the live 30,322-step / 496,795,648-token assistant-only masked SFT at selected
   `3e-4`, under `alpha2-flagship-sft-guard-20260728.service`, preserving `c333bf2` provenance. Its
   first recovery gate passed at step 1,000: held-out loss 1.9429283, zero allocator overflow, exact
   692,528,815-byte remote/mounted checkpoint SHA `9149bc73...`, clean resume through step 1,050, and
   zero guard restarts.
4. Run the frozen chat-side eval and pair analyzer against the completed/mirrored base baseline, do
   the separate human semantic review, then export/verify/publish both HF repos. The exact base export
   already passes stock `LlamaForCausalLM` load, Alpha-vs-Transformers parity (2/2 top-1,
   `6.771e-05` max logit delta), tokenizer parity, and a zero-custom-code CPU `pipeline()` cold load.
