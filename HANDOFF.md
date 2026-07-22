# HANDOFF — alpha2 revival, state as of 2026-07-22 ~20:49 UTC

For the incoming agent. **Read `GOAL.md` first** (repo root) — it is the canonical program: mission,
stage gates G0–G5, budget ledger, standing decisions. This file is the live session-state snapshot and
the exact next steps. Box operating rules live in `/home/ajax/CLAUDE.md`; alpha2 memory in
`~/.claude/projects/-home-ajax/memory/project_alpha2_revival.md`; roadmap block in `/home/ajax/TODO.md`.

---

## ⚠️ LIVE RIGHT NOW — G2 passed; pod is idle, bootstrapped, and billing

- **Pod `d5m7h1v0kr0zd4`**, RTX 3090 community, **$0.22/hr**. SSH:
  `ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p 8865 root@64.119.209.250`.
- **G2 PASSED.** The 5,400-step flagship-shape soak completed cleanly at 20:44 UTC on commit `aca9f97`:
  5,400/5,400 finite rows, 88.47M tokens, literal 6h25m monitoring, p10/median 3,721/3,832 tok/s,
  RSS 681–767MB with negative slope, 34 constant temporary slabs, zero allocator overflow, full 692.5MB
  checkpoint. Every analyzer check is true. Evidence:
  `/mnt/donto-data/alpha-runs/g2-soak-wg64-b16-5400-20260722/{RUN.md,g2-analysis.json}`.
- The pod tree is still at `aca9f97` and is now safe to update. Deploy current master, run the fail-closed
  NVIDIA regression wrapper, then start the contracted G3 Llama pilot; do not leave the paid GPU idle.
- RunPod balance was **$66.736685594** immediately after G2; total account burn was $0.301/hr including
  unrelated stopped volumes. Never delete those unrelated pods. If abandoning this work, terminate this pod with
  `runpodctl remove pod d5m7h1v0kr0zd4`.

## Takeover progress (supersedes stale state later in this file)

- Current functional tree is pushed through **`7171f6d`** before this G2 record update. TypeScript clean; consolidated suite
  **200 pass / 46 GPU-gated skip / 0 fail**. Root `npm test` is pre-existingly broken
  because Turbo runs Vitest in empty packages; use `npm test -w @alpha/tests`.
- NVIDIA gate work, G1, allocator wiring, and post-slab baseline are done and pushed: 46/46 NVIDIA tests;
  G1 1,000 steps with zero NaN; slab profile WG64/pool512; 57.69M-param baseline improved 3,322→3,790
  tok/s (+14.1%). Relevant commits: `e60391e`, `f595708`, `f7730c6`, `32392a5`, `9d7fbc9`, `aca9f97`.
- G4 data gate **passed**. Canonical SFT v2:
  `/mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt`, 511,428/511,428 clean, SHA `ffad0a37…`, exact
  p50/p95/p99/max 657/978/1,014/1,024 tokens, zero over-bound, SODA 4.828%, real assistant mask green.
  Frozen eval: 49 OASST2 validation + 48 Magpie + 3 everyday prompts, 200 QA, 1,500 validation docs;
  exact SFT audit scanned 205,027,527 13-grams and rejected 658/900; final overlap zero. See
  `docs/SFT_CORPUS.md`, `docs/FROZEN_EVAL.md`, and mounted `RUN.md`/manifests.
- G3 pilot launcher is pushed in `scripts/run_g3_pilot.sh`: equal 100,007,936 tokens, 57.69M Llama vs
  58.09M GPT-2 control. Commit `cc7f450` isolates train/validation RNGs across architectures, seeks all
  loader types on resume, writes an immutable pilot contract, and adds `analyze_g3_pair.ts`; `b97a810`
  enforces full-token corpus coverage, and `5e5b913` parameterizes the contracted LR sweep. `da39e8a`
  decouples frequent validation from full optimizer checkpoint cadence and proves the remote-retention
  guard: old remote copies are pruned only after byte-size + SHA-256 agreement with the mounted-drive
  mirror. `867f016` also pins every paid pilot architecture argument explicitly and makes the analyzer
  reject model-config drift; its 6,104-row synthetic contract proof passed. `58fc691` makes the pilots
  safely resumable: exact original contract required, post-checkpoint metric tails preserved+hashed,
  active metrics atomically realigned, and every attempt recorded in `resume-ledger.jsonl`. Do not start
  either pilot until the G2 soak finishes and its artifacts are archived.
- The LR sweep is now proof-gated too: `analyze_lr_sweep.ts` shares the strict pilot validator and selects
  among exactly `{1e-3,2e-3,3e-3}` by the final-three aligned held-out-loss mean (final loss/lower LR are
  deterministic tie-breaks). Its positive and contract-rejection synthetic tests passed in `61c1edb`;
  `59c62dd` additionally requires complete 100-step allocator telemetry through the final pilot row.
- `run_flagship_pretrain.sh` (`f6c590e`) consumes only a selected contracted LR and launches exactly
  1,000,013,824 tokens over the verified three-shard manifest, with the explicit Llama/AdamW profile,
  independent eval/checkpoint cadence, immutable contract, and safe resume ledger. Contract-only positive
  and wrong-LR rejection proofs passed without launching training.
- Base→SFT initialization is no longer conflated with resume: `--initCheckpoint` (`55c86db`) validates
  and restores weights only, resets the declared RNG, and starts a fresh optimizer/schedule at step zero;
  it is mutually exclusive with continuation `--resume` and has bit-identical parameter proof at LR zero.
- Token caches are artifact-bound and crash-safe in `45bfe60`: exact tokenizer SHA in the key, checked
  chunked I/O, source mtime+size header, fsync+atomic rename, and automatic truncated-cache recovery.
- Checkpoint compatibility in `6b460e4` covers semantic architecture, not just dimensions: norm type,
  positional encoding, RoPE theta, embedding tying, and soft-cap mismatches all fail closed.
- Flagship SFT is contracted in `7636ad2`: `verify_flagship_sft_inputs.ts` independently streams the
  511,428-row corpus, reconciles both passed audits, derives the exact 485,150/26,278 split, and verifies
  every base-checkpoint parameter byte is present and finite. `run_flagship_sft.sh` admits only the
  `{1e-4,3e-4,1e-3}` sweep, launches exactly one 30,322-step assistant-only epoch, separates weight-only
  initialization from continuation resume, and records immutable hashes. Real corpus + step-100 fixture
  proof passed; wrong-base-step rejection passed; TypeScript and 200/46 consolidated gates stayed green.
- Long-run checkpoint growth is bounded in `99a9116`. Use matching
  `REMOTE_KEEP_CHECKPOINTS=3 LOCAL_KEEP_CHECKPOINTS=3`: remote deletion still requires local byte+SHA
  proof, then local pruning keeps the newest three and fsyncs before/after deletion records (including
  the removed hash) to `checkpoint-prune-ledger.jsonl`. Counts below three or mismatched policies fail
  before SSH. The isolated six-checkpoint fixture retained 4–6, ledgered+removed 1–3, and was idempotent.
- Frozen base-vs-chat evaluation is tamper-evident in `863427f`: v2 summaries hash both detailed JSONL
  outputs and bind EOS/user control IDs; `analyze_frozen_eval_pair.ts` recomputes all 100 chat + 200 QA
  flags/scores, requires exact 61,036/30,322-step checkpoints and identical frozen inputs/case order, and
  enforces the ≥95 structural / zero-loop machine bar. Its PASS explicitly leaves conversational
  coherence to separate semantic review. Full synthetic pair passed; altered output hash was rejected.
- Post-G2 NVIDIA regression is fail-closed in `1019b9b`. Run
  `scripts/run_nvidia_gates.sh /workspace/alpha2/runs/nvidia-gate-<commit>` after deploying current
  master; it requires vendor `0x10de` and the exact two files / 46 unique assertions / 46 passed / zero
  skipped-failed-todo, then hashes the Vitest JSON into `gate-summary.json`. The real local all-skipped
  report was rejected, a synthetic 46/46 report passed, and non-NVIDIA preflight stopped before tests.
- SFT LR selection is executable rather than aspirational in `b24c18a`: three sequential
  `run_sft_lr_pilot.sh` runs each consume exactly 2,000 steps / 32,768,000 padded tokens with eight
  aligned validations and the identical verified corpus/audits/tokenizer/base. The selector requires
  complete finite runs, zero allocator overflow, full checkpoints, identical inputs+commit, and ranks
  final-three held-out loss. `run_flagship_sft.sh` now refuses to start without the matching report and
  verifies its selected LR plus every input hash. Positive and mismatch synthetic proofs passed.
- Immediate order: (1) commit/push this G2 PASS record; (2) fast-forward the idle pod to current master,
  build, and certify the exact tree with `run_nvidia_gates.sh`; (3) sync+hash the canonical G3 corpus and
  launch the Llama half with the verified mirror/retention guard; (4) run the GPT-2 half sequentially and
  compare with `analyze_g3_pair.ts`; (5) run the three-way LR sweep; (6) resumable flagship,
  SFT, frozen eval, HF upload. Host disks are unexpectedly full (root 91%, data 87% at 16:24); avoid
  unbounded artifacts and do not destructively clean without resolving exact targets.
  The analyzer also requires the final full model+AdamW checkpoint to be 650–750 MiB (`ddd9bd3`), so a
  nonempty/truncated placeholder cannot satisfy G2.

## Mission in one line

Train a small chatty model **entirely with Alpha's own from-scratch stack** (TS tensor lib, tape
autograd, hand-generated SPIR-V, Helios Vulkan backend — GPU-resident, no PyTorch/CUDA training) on
RunPod, publish to Hugging Face as a **standard zero-custom-code `LlamaForCausalLM`** repo
(`ajaxdavis`, HF auth on box verified WRITE-capable). Operator soul constraint (2026-07-22): every
training FLOP through Alpha's own code. User also directed: **all box-side code work before GPU spend**
— that is now DONE; user has explicitly said to move to GPU ("this box can't handle it" — do NOT run
more CPU training on the box).

## State of the repo (github.com/thomasdavis/alpha2, master, all pushed)

Working tree CLEAN at `84c110c`. Key commits this program (chronological):
- `9524598` GOAL.md + proven RunPod/Vulkan bootstrap (`scripts/runpod_bootstrap.sh`, `docs/RUNPOD.md`)
- `59d79da` **G0 PASSED**: Helios trained on a RunPod 3090 (60 steps, loss 7.28→7.05, 0 NaN, ~40K tok/s
  at 1.33M params, DGC+BDA+coop active). Artifacts: `/mnt/donto-data/alpha-runs/g0-smoke-20260722/`
- `aea174c` deps → latest (TS 7.0.2, vitest 4, Next 16, ai v7, effect 3.22) + 4 known-bug fixes
  (lmHead no-decay name, `--vocabSize` silently ignored, fp16 auto-enable trap, train-nanochat lr
  6e-4→3e-4) + secrets scrub (Discord webhook REVOKED via DELETE 204; `movies/symbio-film/.env`
  untracked — **ElevenLabs key still needs USER dashboard rotation**, it's in public git history)
- `0222fb3` npm audit 5→1-low (overrides sharp/postcss/esbuild; remaining 1 is Windows-only dev-server)
- `9b63685` **Stage-1 gradcheck harness** (see below) + REAL bug fix: `cpu_ref.sum(axis, keepdims=true)`
  was wrong on non-last axes → corrupted broadcast backward grads. Found by the harness day one.
- `fcfa83a` corpus builders (`scripts/build_pretrain_corpus.py`, `scripts/build_sft_corpus.py`)
- `b3ffe90` **Stages 3–4 box-side** (the big one, ~3,200 lines — see below)
- `84c110c` e2e script self-containment fix + final-tree golden numbers

### Test/verification state (all on the FINAL committed tree)
- `nice -n19 npx tsc -b` from root: clean. Full turbo build: 19/19.
- `packages/tests`: **178 passed / 44 GPU-gated skips / 0 failed** (~80–150s wall on the loaded box).
  The 44 skips are `parity-helios.test.ts` (36) + `gpu-perf.test.ts` — they gate on NVIDIA vendorId
  0x10de and have **NEVER run on real NVIDIA hardware**. Running them is the top next step.
- **Golden-token gate (G3 export half) PASSED on final tree**: tiny Llama-form model trained on
  cpu_ref → `alpha export-hf` → loaded by `transformers` (fp32, no trust_remote_code):
  **75/75 top-1 = 100%, max |Δlogit| 1.07e-06** (threshold 1e-3), tokenizer parity 4/4 prompts exact.
  Reproduce: `bash scripts/e2e_hf_export.sh` (self-contained; caches its checkpoint under
  `/mnt/donto-data/alpha-runs/g3-e2e/`). **But do NOT re-run CPU training on the box — user said stop.**
- Byte-BPE exporter cross-verified vs Python `tokenizers` on 9,822 real corpus docs: 100% id agreement.

### What Stages 3–4 added (commit `b3ffe90`)
All config-gated; legacy GPT-2-style configs bit-for-bit unchanged.
- **Arch**: `rmsNorm` (+fused backward) and `rope` ops — cpu_ref + autograd + Helios SPIR-V kernels.
  RoPE is EXACTLY HF `rotate_half` (half-split, `inv_freq=θ^(-2i/D)`) so export needs no permutation;
  backward = rotation by −angle (reuses forward kernel with negated sin). Tied embeddings via
  `lmHead === wte` object identity. `ModelConfig`: `normType`/`posEnc`/`ropeTheta`/`tieEmbeddings`.
  softCap defaults OFF under rope. New domain **`alpha_llama`** = 16L/512d/8H swiglu(1408) rmsnorm rope
  tied, block 1024, tokenizer `bpe-byte-12k` (~60M params — the flagship shape).
- **Tokenizer**: `ByteBpeTokenizer` (`packages/tokenizers/src/byte_bpe.ts`) — 256-byte base (exact GPT-2
  bytes_to_unicode), GPT-2 split regex, lossless decode on anything, atomic chat specials
  `<|user|>` `<|assistant|>` `<|end_of_text|>` (ids 256/257/258). Registry: `bpe-byte-12k`, `bpe-byte-4k`.
  HF exporter (`export_hf.ts` + `alpha tokenizer export-hf`) emits tokenizer.json/tokenizer_config.json/
  chat_template.jinja.
- **SFT loss masking**: `DataBatch.lossMask` + `crossEntropyMasked` (cpu_ref + fused Helios masked-CE
  kernels), SFT loader mode (one conversation per row, assistant-span-only mask, `--sft` on train cmd),
  trainer threads mask through grad-accum + eval.
- **HF export**: `packages/train/src/hf_export.ts` (spec-exact TS safetensors writer; ALPH→Llama state
  dict: wqkv split to q/k/v, fc_gate/up/proj→gate/up/down, NO transposes — `[out,in]` matches nn.Linear;
  omits lm_head when tied) + `alpha export-hf` + `alpha logits` CLI + `scripts/verify_hf_export.py`
  (golden verifier; py deps live in the uv venv at `/mnt/donto-data/alpha-corpora/.venv`: torch-cpu,
  transformers, safetensors, tokenizers, pyarrow).
- **Inference engine** (`packages/inference`): now supports rope/rmsnorm/tied (crash on Llama-form
  checkpoints fixed) + inference-parity tests. This unblocks the HF Space / `alpha sample` fast path.
- **Adversarial-review fixes (measured, not guessed)**:
  - P0: Helios masked-CE kernels had Out/Mask bindings swapped → GPU SFT would have trained on
    garbage SILENTLY (loss exactly 0). Fixed kernel-side (Out = last binding); documented in
    `kernels/nn.ts` next to `ce_fwd_masked`.
  - P1: flash-attention q/k/v used a PLAIN reshape `[B,T,nH*hd]→[B*nH,T,hd]` that scrambles
    (batch,head) rows for nHead>1 — PRE-EXISTING bug affecting all prior multi-head flash training;
    now reshape→transpose(1,2)→reshape head-major (see `gpt.ts` "[defect P1]" comment). RoPE positions
    on the flash path are now correct.
  - P2: shared-memory reduction races in CE/layerNorm/rmsNorm kernels — trailing ControlBarriers added
    (12 sites). Masked by NVIDIA warp lockstep; exposed on relaxed schedulers (rmsNorm dx diverged ~4e3
    on llvmpipe without it).
  - P3 (= the inference-engine fix above).

### Stage-1 harness (commit `9b63685`) — how correctness is enforced
`packages/tests/src/`: `gradcheck-ops.test.ts` (central-difference FD checks for EVERY op the model
uses; reusable `checkGrad`), `gradcheck-model.test.ts` (whole tiny-GPT gradchecks across
swiglu/gelu/universal/kan_spline AND the Llama-form config; top-|grad| element sampling; dead-param +
determinism + checkpoint-bitwise invariants), `optimizer-reference.test.ts` (AdamW vs independent
reference <1e-6), `parity-helios.test.ts` (GPU-gated CPU↔Helios parity: per-op fwd/bwd, tiny-GPT logits/
grads/AdamW-step, 100-step zero-NaN loop, f16 casts, rmsNorm/rope parity, tied-model loop, masked-CE).
**Any new op MUST get: cpu_ref impl + autograd backward + Helios kernel + FD gradcheck + parity test.**
Every check was proven load-bearing by temporary fault injection.

## Training data (READY, on the data disk)

- Pretrain: `/mnt/donto-data/alpha-corpora/pretrain-text/` — 6 shards ≤2GB, 11.7GB, ~3.0B est tokens,
  1.86M docs, `<|end_of_text|>`-delimited. Source: 4 parquet shards of
  `HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` (kept in
  `premix-shuffled/`; 96 more shards available upstream if more tokens needed). All six outputs are
  sealed and re-verified by `pretrain-text/MANIFEST.sha256`; see the adjacent `RUN.md`. The minimum
  flagship uses `flagship-1b-manifest.json` (first three shards, 5,976,889,749 verified bytes) through
  the deterministic sharded loader in `28c6506`, avoiding both data repetition and giant-buffer limits.
- SFT: `/mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt` — 511,428 structurally clean,
  tokenizer-bounded conversations; SHA-256 `ffad0a376c7eac2e0ec91f0901ec1ff87cba67cc298222828ce3df1a3e60b3fb`.
  The previous unbounded version is preserved under that corpus directory's `history/`.
- Tokenizer: durable canonical artifact
  `/mnt/donto-data/alpha-runs/tokenizers-20260722/g2-bpe-byte-12k.json`; SHA-256
  `c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24`. It was built on the pod from
  the 128MB pretrain slice, used by G2, then mirrored and local/remote hash-verified. See its `RUN.md`.

## Infra / credentials (all verified working this session)

- **RunPod**: `runpodctl` 2.6.1 configured (`~/.runpod/config.toml`); GraphQL with the same key.
  SSH key `~/.runpod/ssh/runpodctl-ssh-key`. Community RTX 3090 = proven Vulkan host class.
  Prices (community): A5000 $0.16 · 3090 $0.22 · A40 $0.30-0.35 · 4090 $0.34.
- **Vulkan-on-RunPod recipe (PROVEN)**: `scripts/runpod_bootstrap.sh` — driver-matched NVIDIA `.run`
  userspace install (`--no-kernel-modules`, kmod stubs) + **EGL headless ICD** + `VK_ICD_FILENAMES` +
  ctypes probe. Full runbook `docs/RUNPOD.md`. If the probe fails on a host: TERMINATE AND REDEPLOY,
  never debug a bad host. Community-host egress is a lottery: apt (port 80) and github may be dead;
  nodejs.org + download.nvidia.com (443) worked everywhere so far. Deploy code via **rsync from the
  box** if git clone fails (`--exclude=.git --exclude=.next --exclude=.turbo`; sync `packages apps`
  first if in a hurry — node_modules is 1GB and rsyncs alphabetically; full sync ~30-45 min under box
  I/O load).
- **Hugging Face**: `hf` CLI authed as `ajaxdavis` (write verified by probe create+delete).
- **Box rules**: shared multi-tenant box — EVERYTHING niced (`nice -n19`, `ionice -c3` for I/O);
  temp files under `$CLAUDE_JOB_DIR/tmp`, NOT bare /tmp; research artifacts under /mnt/donto-data;
  **no more CPU training on the box (user directive)**. lint-staged pre-commit runs a full turbo build
  and TIMES OUT under load — if you've manually verified build+tests, `git commit --no-verify` and say
  so in the message. Commit + push often (user directive; memory `feedback_commit_push_often`).
- Node runtime ONLY for Helios (`node --expose-gc apps/cli/dist/main.js`); the bun compiled binary has
  a known vkCreateInstance failure. Always `--fp16=false` posture (fp16 auto-enable removed, but be
  explicit); `HELIOS_DISABLE_COOP_MAT=1` for training stability per docs.

## NEXT STEPS (in order — the pod is waiting)

1. **Bootstrap the live pod** (`d5m7h1v0kr0zd4`, 64.119.209.250:8865):
   `scp scripts/runpod_bootstrap.sh` → run it → expect `vkCreateInstance OK, 1 device(s)`.
2. **Deploy code**: try `git clone https://github.com/thomasdavis/alpha2 && npm install` on the pod
   (repo is public + fully pushed); if egress is broken, rsync from
   `/mnt/donto-data/workspace/alpha2/` incl. node_modules. Then `nice npm run build` (or
   `npx turbo build --filter=@alpha/cli --filter=@alpha/tests` — much faster, skips the web app)
   and `node packages/helios/native/build.mjs` if the box-built addon doesn't load (`ldd` it).
3. **Run the GPU gates** (never executed on NVIDIA — this is the payoff of all the box work):
   - `cd packages/tests && npx vitest run parity-helios gpu-perf` → **all 44 must pass.**
     Watch specifically: masked-CE parity (P0 fix), rmsNorm/rope parity, the f16 cast tests,
     the tied-model 20-step loop, flash-vs-standard after the P1 relayout.
   - **G1 pilot**: 1,000 steps, ~10M params (e.g. 6L/256d/4H alpha_llama-style, bpe-byte-4k) on a
     pretrain shard slice, f32, helios — **gate: ZERO non-finite gradient steps** (the old 2-7% NaN
     era must be provably over; the SwiGLU/Helios interaction root-cause may surface here — if NaNs
     appear, bisect with the parity suite, do NOT mask with spike-skip).
   - **G2 baseline measurement**: 100-200 steps at the flagship `alpha_llama` shape (16L/512d, block
     1024, batch to fit 24GB) — record tok/s + live-alloc telemetry. Expect ~1K tok/s (allocator-bound).
     This anchors Stage 2.
   - Pull all runs/logs to `/mnt/donto-data/alpha-runs/` (box-side puller loop in docs/RUNPOD.md),
     then TERMINATE the pod. Update GOAL.md gates + ledger + commit.
4. **Stage 2 (the throughput unlock — biggest remaining engineering)**: wire device-local slab
   (TS never passes `temporary=1`; native slab code exists in `helios_vk.c` but is bypassed →
   every device tensor is an individual vkAllocateMemory → GC storms ≥192d). Gate G2:
   **≥3,000 tok/s sustained at flagship shape + 6h soak, zero allocator crashes**. Budget math:
   1B tokens @3K tok/s ≈ 93 GPU-h ≈ $20 on the 3090.
5. **Stage 5 flagship** per GOAL.md: lr sweep {1e-3, 2e-3, 3e-3} at 100M-token pilot scale (the old
   3e-4 lore predates the bug fixes — re-derive), then ~60M pretrain on 1-3B tokens (budget-gated by
   measured tok/s), then masked SFT on the built corpus, frozen evals (GOAL D3 chat bar).
6. **Stage 6 ship**: `alpha export-hf` the flagship → `hf upload ajaxdavis/alpha-60m-base` +
   `-chat` → `pipeline()` cold-load verify → model cards with honest evals + data licenses
   (ODC-BY/Apache-2.0/CC-BY-4.0 attribution). GGUF is a stretch (needs the `get_vocab_base_pre`
   patch, pre name "gpt-2").

## Known gaps / watch-outs

- **flash-attention on GPU**: P1 relayout is committed but flash parity has never run on NVIDIA.
  The parity suite covers it; if flash still diverges from standard on the 3090, train with the
  standard path (flash is a perf optimization) and file it for Stage 2.
- The trainer's in-loop sample subprocess uses cpu_ref — fine (tiny), but `--sampleInterval` large
  keeps pod CPU free.
- `alpha_llama` lr 3e-4 in domains.ts is a PLACEHOLDER — Stage 5 sweeps it.
- Data loader holds the whole tokenized corpus in RAM (Int32 = 4 bytes/token): 1B tokens = 4GB,
  3B = 12GB — check pod RAM at create (3090 hosts vary; ask for ≥32GB vCPU RAM if running 3B).
- `MAX_STRING_BYTES` 30MB / 10MB-chunk tokenization is handled by `loadAndTokenize` — corpus shards
  are already ≤2GB each, fine.
- Eval set (GOAL Stage-4 item) is NOT yet frozen: before the flagship run, build the fixed
  100-chat-prompt + 200-question + repetition/EOS suites (smol-smoltalk test split is reserved for
  this). Don't let benchmark data into training mixes.
- The box `.venv` for python verify work: `/mnt/donto-data/alpha-corpora/.venv` (activate then run
  `scripts/verify_hf_export.py` / `verify_tokenizer_export.py`).
- OUTSTANDING USER ACTIONS (already flagged, don't nag): rotate ElevenLabs key; decide fate of the
  4 stopped migmaq pods.

## The one-paragraph story so far

In one day the project went from a 4-month-dormant repo to: proven Vulkan-on-RunPod (G0: Helios trained
on a $0.22/hr 3090, zero NaN), a fully modernized toolchain (TS7/vitest4/Next16), a fault-injection-
proven gradient-checking harness that immediately caught a real broadcast-gradient bug plus a GPU-SFT-
would-have-been-garbage kernel-binding bug and a scrambled-flash-attention layout bug, a Llama-form
architecture (RoPE/RMSNorm/tied/byte-BPE) whose exports load in stock `transformers` at 100% top-1
agreement, assistant-only loss masking, 3B tokens of pretrain text + 457K SFT conversations staged,
and ~$0.4 of the $70 GPU budget spent. The remaining path to a shipped model is: GPU gates → slab
allocator throughput work → flagship pretrain+SFT → `hf upload`.
