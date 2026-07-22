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
  priority and all 5,976,889,749 remote bytes match the immutable manifest hashes.

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
      base/chat steps and identical inputs/case order, and enforce the machine-verifiable D3 threshold
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
  `run_flagship_pretrain.sh` (`f6c590e`) admits only an LR from the contracted sweep and fixes the exact
  61,036-step / 1,000,013,824-token architecture, optimizer, eval, checkpoint, manifest, tokenizer,
  commit, and resume contract.
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
  The choice itself is contracted in `b24c18a`: three 2,000-step / 32,768,000-token pilots over
  `{1e-4,3e-4,1e-3}` share exact inputs and eight aligned validations; `analyze_sft_lr_sweep.ts` chooses
  the lowest final-three held-out mean. The one-epoch launcher now requires and verifies that selector
  report, including its commit and all six input hashes, rather than accepting any allowed number.
- **Ops discipline** (box CLAUDE.md rules apply): verify-it-actually-works — measure real tok/s from
  metrics deltas not logs; watchdog terminates any pod whose checkpoint stream stalls 30 min; every
  run resumable (`--resume`); no fire-and-forget. `99a9116` bounds ~693MB checkpoint growth with matched
  three-copy remote/local retention: remote pruning requires a byte+SHA mirror proof, and each subsequent
  local deletion is itself hash-ledgered before and after unlink. The policy is opt-in and rejects fewer
  than three recovery points.
- **Gate G5 = D3 chat bar.** If quality is word-salad at the bar, we do NOT ship a chat model; we ship
  the base model with an honest card and the ledger records what a bigger budget would change.

### Stage 6 — Ship (box only, $0 GPU)
- [ ] TS **safetensors writer** (~25 lines, spec is trivial: u64 header-len + JSON + raw LE f32) with
      `__metadata__ {"format":"pt"}`; round-trip verified against Python `safetensors` on the box.
- [ ] `config.json` (llama; explicit `num_key_value_heads`, `head_dim`, `rms_norm_eps=1e-5`, flat
      `rope_theta`, `tie_word_embeddings: true` + **omit lm_head from the file** — the #1 silent-garbage
      pitfall), `generation_config.json`, `chat_template.jinja` with `{% generation %}` markers.
- [ ] Model cards: from-scratch story, stack description, data mix + licenses (ODC-BY / Apache-2.0 /
      CC-BY-4.0 attribution), full eval table incl. failures, energy/cost actuals from the ledger.
- [ ] `hf upload` both repos; cold-load verify via `pipeline()` on the box; tag `alpha2` release commit.
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

## 8. Immediate next actions (current 2026-07-22)

1. Let the healthy G3 Llama pilot and its persistent guard carry all 6,104 finite rows and the final
   checkpoint onto the mounted drive.
2. Run GPT-2 sequentially on exact commit `c95f81b` with the same input/tokenizer/LR, then evaluate the
   100,007,936-token pair with `analyze_g3_pair.ts`. The golden export half is already 75/75 top-1 with
   max logit delta 1.07e-06. Do not pull a newer source commit between the two architecture runs.
3. Run the three-way 100M-token LR sweep, choose in the ledger, then begin the resumable flagship pretrain
   with a verified box-side checkpoint puller.
