# Alpha and Helios — comprehensive handoff to the next agent

**Frozen at:** 2026-08-03 11:49 UTC  
**Repository:** `/mnt/donto-data/workspace/alpha2`  
**Branch:** `agent/alpha-chat-repair-v2-closeout`  
**Implementation baseline:** `b72d03886076dde617389eb053af190cf681e791` (`Script the physical column reduction sweep`); the handoff itself is a later documentation-only commit  
**Remote:** `https://github.com/thomasdavis/alpha2`  
**Intended final worktree state:** clean, with local branch and `origin/agent/alpha-chat-repair-v2-closeout` at the same handoff commit  
**Billing state at handoff:** `runpodctl pod list` returned `[]`; no Alpha RunPod is running  
**Disk state at handoff:** `/mnt/donto-data` has 93 GiB free; this checkout occupies 5.3 GiB

This is the self-contained transfer document for another agent model. The root
[`HANDOFF.md`](../../HANDOFF.md) remains the full chronological archive, but it contains historical states that
were valid at different times. Use this document for the current decision boundary and then follow its links
when deeper evidence is needed.

No access token, API key, webhook URL, private endpoint credential, or Hugging Face credential is written here.
Secrets are local and untracked. Never paste them into a commit, log, mounted research archive, Discord message,
or public artifact.

---

## 1. Sixty-second state

The product goal is to finish the original Alpha project as a small-enough-for-one-GPU model that is genuinely
pleasant and effective in ordinary conversation. It should start answers reliably, understand what the user is
trying to say, respond contingently rather than by slogan or template, sustain a discussion, and eventually be
especially good at language, linguistics, ontology, philosophy, evidence, conceptual distinctions, and the
structure of knowledge. It does not need to memorize every fact; retrieval can supply exact and changing facts.

The current public model is an honest but weak intermediate checkpoint. It is structurally more chatty than the
failed terminal SFT model, but it is semantically shallow, repetitive, often wrong, and still reports
`quality_gate=FAIL`. Several later chat-repair and same-data experiments failed to improve it. Do not publish a
new model merely because it answers every prompt or has lower teacher-forced loss.

The current engineering strategy is therefore:

1. make the native Helios training engine fast and trustworthy enough for an affordable full foundation run;
2. train the frozen foundation candidate from scratch on the verified packed pretraining corpus;
3. perform conversational distillation/post-training selected by untouched free conversation;
4. publish a new Hugging Face and BLAH version only when the model is behaviorally better;
5. fit the requested Jacobian Lens bundle only after the exact winning checkpoint is immutable.

Helios has already moved from a historical 3,579 tokens/s to a selected median 7,253.8 tokens/s on an RTX 4090
for the exact foundation shape, a 102.7% end-to-end gain. The next candidate is a row-parallel RMSNorm
`column_sum` reduction. It is locally correct, physically tunable at 4/8/16 row lanes, disabled by default, and
has a complete mirrored physical-device sweep script. **It has not yet been benchmarked on a physical GPU and
there is no speed claim.**

The full foundation run has **not** begun. Verified caches and the completed learning-rate pilot are not a
trained model.

---

## 2. Operator intent and product north star

### 2.1 What the user actually wants

Alpha is first a conversational model, not an ontology database, benchmark artifact, kernel collection, or
paper. The desired experience is closer to an intelligent, curious interlocutor who can:

- answer directly and naturally;
- infer intent from imperfect wording;
- ask only genuinely useful clarifying questions;
- maintain momentum without canned follow-up questions;
- discuss ambiguity, reference, roles, events, evidence, time, identity, part/whole relations, purposes, and
  competing conceptualizations;
- challenge a premise when that helps, while remaining collaborative;
- preserve a locally established meaning across turns;
- revise the affected idea after a counterexample without throwing away everything else;
- use search or supplied evidence for exact facts rather than bluffing.

“Chatty” does not mean verbose. It means responsive, contingent, adaptive, present, and worth continuing to talk
to. A long mini-essay that ignores the conversational move is a failure. A single attractive sample is not proof
of improvement.

### 2.2 What must not displace the product

AlphaCorpus, Inferential Conceptual Pacts, Predicate Birth, Alpha Joints, Donto integration, synthetic-data
infrastructure, the public `/corpus` browser, and Jacobian Lens work are valuable adjacent programs. They are not
the present model-training north star. Do not restart the corpus-generation program merely because its archived
PRDs are detailed.

Synthetic data will eventually be a major part of post-training. It should be richly categorized and retained in
a scientific SQLite ledger with generation, review, rejection, relation, rendering, and exposure lineage. But the
immediate model path is foundation competence and ordinary conversational quality. Formal semantic structure is
researcher-side scaffolding; it should not make Alpha speak in JSON, ontology notation, database fields, or rubric
labels.

### 2.3 Model-size language

Do not use parameter count as the novelty claim or a benchmark to optimize for its own sake. The operator wants
the model to remain practical on one rented GPU. Exact parameter counts still belong in reproducibility records
and resource planning.

---

## 3. Current authority and goal-manager state

The durable goal recorded for this thread is:

> Finish Alpha as a genuinely chatty, effective conversational model by diagnosing frozen-evaluation failures,
> executing evidence-gated one-GPU training experiments, selecting by untouched conversational behavior,
> preserving complete recovery evidence, and publishing the best honest checkpoint.

At handoff, the goal manager reports this goal as **blocked**, not active. An attempt to create the later
“34-hour Helios performance campaign” goal was rejected because the earlier goal remained unfinished. This is a
goal-manager state, not evidence that the technical program is impossible. The user may need to resume or replace
the goal in the product UI when the next agent takes over. Do not silently call the old goal complete.

The latest operator direction before this handoff was to keep the RunPod off and continue Helios performance
work, save all research and benchmarks to disk, show readable time-breakdown tables, and report meaningful or
interesting findings to Discord. If a physical-only discriminator is ready, create a bounded GPU environment,
run the complete predeclared comparison, preserve and verify the artifacts, then delete the exact pod. Do not
leave a paid machine idle.

---

## 4. Exact repository and live-service state

### 4.1 Git

```text
checkout  /mnt/donto-data/workspace/alpha2
branch    agent/alpha-chat-repair-v2-closeout
head      b72d03886076dde617389eb053af190cf681e791
remote    https://github.com/thomasdavis/alpha2
status    clean
```

Verify, rather than assuming this remains current:

```bash
cd /mnt/donto-data/workspace/alpha2
git status --short
git branch --show-current
git rev-parse HEAD
git log -12 --oneline --decorate
git remote -v
```

The most recent commits are:

| Commit | Meaning |
|---|---|
| `b72d038` | Added the immutable, mirrored physical column-reduction sweep |
| `cc21c92` | Added reproducible Markdown/JSON Helios profile summarization |
| `11d5ccd` | Made row-parallel reduction physically tunable at 4/8/16 lanes |
| `2bb3e41` | Added the opt-in row-parallel column-reduction candidate |
| `48fab24` | Hardened cooperative-forward diagnostics and provenance capture |
| `00bc9a5` | Exercised the production cooperative tile in the oracle |
| `1b94488` | Added production-shape cooperative GEMM gates |
| `8ab812d` | Required mounted Helios evidence preservation |
| `93c18d0` | Recorded K32 preflight and verified pod shutdown |
| `2ca8692` | Added experimental K32 transposed-B GEMM |
| `3abc1a0` | Recorded the selected transposed-A coalescing evidence |
| `028e9b3` | Added coalesced transposed-A register-blocked loads |

### 4.2 RunPod and host

At 2026-08-03 11:49 UTC:

```text
runpodctl pod list => []
/mnt/donto-data    => 984G total, 871G used, 93G available, 91% used
repository         => 5.3G
```

Recheck both before any expensive operation:

```bash
runpodctl pod list
df -h /mnt/donto-data
du -sh /mnt/donto-data/workspace/alpha2
free -h
```

The project-specific instruction is to pause for review before this project adds more than 15 GiB. This is not a
license to discard evidence. Keep new artifacts on `/mnt/donto-data`, compress checkpoints losslessly after
verification, and ask the operator to review storage if the new project slice approaches that amount.

### 4.3 Public model and serving snapshot

The public selected model remains:

```text
Hugging Face repo     ajaxdavis/alpha-60m-chat
HF immutable revision ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0
weights SHA-256        a5214ebad501b8bd3b09f7552c0db67417d18c3b66432f66f847de0e723dd688
native selected step   1,200
native SHA-256         399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec
quality gate           FAIL
```

The public BLAH record verified at handoff is:

```text
model ID        gtp2y4-YOuje_yrdvZlxsw
name            Alpha 60M Chat
inference model ajaxdavis/alpha-60m-chat
```

The open BLAH log endpoint reported 1,275 calls, 26 errors, and 8,670 ms mean latency at the handoff snapshot.
Those are volatile service counters, not a quality score. Read raw logs and judge reasoning before diagnosing a
model change. The BLAH API key is deliberately absent from this document.

The local CPU backend was active and returned:

```json
{
  "status": "ok",
  "model": "ajaxdavis/alpha-60m-chat",
  "parameters": 57688576,
  "checkpoint_step": 1200,
  "quality_gate": "FAIL"
}
```

Safe live checks:

```bash
curl -fsS https://huggingface.co/api/models/ajaxdavis/alpha-60m-chat | jq '{id,sha,lastModified}'
curl -fsS https://donto.org/alpha-60m/health | jq .
curl -fsS https://evals.blah.dev/api/v1/models/gtp2y4-YOuje_yrdvZlxsw | jq '{id,name,inference_uri}'
curl -fsS 'https://evals.blah.dev/api/v1/models/gtp2y4-YOuje_yrdvZlxsw/logs?limit=50&stats=true' > /tmp/alpha-logs.json
```

Every genuinely updated checkpoint gets a new versioned BLAH model record. Never repoint a previously evaluated
BLAH record to different weights or runtime behavior. Do not publish an engine-only optimization as a new model.

---

## 5. Honest model-quality history

### 5.1 The terminal SFT failure

The original terminal SFT trained without numerical failure but was not a useful chat model:

| Measure | Result |
|---|---:|
| Frozen chat prompts | 100 |
| Empty responses | 92 |
| EOS terminations | 94 |
| Degenerate loops | 6 |
| Structural passes | 2 |
| Blinded semantic verdict | 0 PASS / 0 BORDERLINE / 100 FAIL |
| Closed-book QA | 0/200 exact, 0 contained |

This established the central measurement lesson: token-averaged teacher-forced loss can hide catastrophic
response-initiation behavior.

### 5.2 The selected corrective checkpoint

The step-1,200 corrective checkpoint repaired much of response initiation and became the current public model.
It is still only an intermediate artifact:

| Measure | Result |
|---|---:|
| Selected native checkpoint | step 1,200 |
| Frozen structural result | 55/100 |
| Nonempty | 70/100 |
| Loop flags | 31 |
| QA | 0/200 |
| Development characterization | structurally chatty, semantically immature |
| Release gate | FAIL |

The archive source of truth is a native ALPH checkpoint with optimizer/RNG/tokenizer state, not the public
Safetensors export.

### 5.3 Later negative results

- Repair v2 made all selector responses nonempty, but both continuation and clean-base arms increased repetition
  on the fair 69-prompt `generationEligible` subset. No v2 checkpoint was selected.
- Repair v3 was executed and rejected for release. Do not mistake the earlier “local preflight” language in old
  chronological sections for current state.
- V8–V12 experiments established more reliable response production but did not produce a robust semantic winner.
- V11 generated all development replies but increased repetition, lost its blinded comparison, and scored below
  the earlier Alpha on BLAH.
- The same-data recipe control did not establish the desired conversational quality.

When comparing chat checkpoints, use the baseline-eligible population. Nonempty output is not conversational
competence, and output-format mismatches must be separated from model-quality failures.

Detailed evidence:

- [`CHAT-REPAIR-2026-07-31.md`](CHAT-REPAIR-2026-07-31.md)
- [`CHAT-REPAIR-V2-2026-07-31.md`](CHAT-REPAIR-V2-2026-07-31.md)
- [`CHAT-REPAIR-V2-MECHANISM-ANALYSIS-2026-08-01.md`](CHAT-REPAIR-V2-MECHANISM-ANALYSIS-2026-08-01.md)
- [`CHAT-FOUNDATIONS-V11-OUTCOME.md`](CHAT-FOUNDATIONS-V11-OUTCOME.md)
- [`CHAT-RECIPE-V12-LR1E3-OUTCOME.md`](CHAT-RECIPE-V12-LR1E3-OUTCOME.md)
- [`CHAT-RECIPE-V12-LR3E4-OUTCOME.md`](CHAT-RECIPE-V12-LR3E4-OUTCOME.md)
- [`BLAH-BASELINE-AUDIT-2026-08-02.md`](BLAH-BASELINE-AUDIT-2026-08-02.md)

---

## 6. Current model-development strategy

The evidence supports a foundation-first path rather than more blind SFT on the weak base:

1. Train a stronger small foundation from scratch on the verified pretraining corpus.
2. Distill general conversational behavior and language understanding from strong teachers.
3. Post-train for answer initiation, stopping, contingency, instruction following, multi-turn state, and ordinary
   conversation before adding a dense specialist curriculum.
4. Add synthetic linguistic, pragmatic, ontological, philosophical, evidence, and conceptual-pact data as
   linked natural conversations rather than formal model-visible records.
5. Select checkpoints by free generation on untouched conversations and human comparative judgment.
6. Use BLAH logs and versioned evals as diagnosis, not as the sole optimizer.

This path is not a claim that synthetic data is unimportant. The operator expects synthetic-data generation and
review to be roughly half the eventual project. It is a sequencing decision: a model that lacks foundational
language competence or cannot initiate a response cannot reveal whether the specialist curriculum worked.

---

## 7. Frozen foundation candidate

The exact current contract is:

| Field | Value |
|---|---:|
| Layers | 18 |
| Hidden width | 640 |
| Attention heads | 10 x 64 dimensions |
| SwiGLU FFN width | 1,728 |
| Vocabulary | 12,288 byte-BPE tokens |
| Context | 1,024 tokens |
| Tied embeddings | yes |
| Norm / position | RMSNorm / RoPE |
| Batch | 24 |
| Gradient accumulation | 1 |
| Parameters | 97,098,880 |
| Precision | FP32 |
| Optimizer | AdamW |
| Peak LR | 0.002 |
| Minimum LR | 0.0002 |
| Warmup | 790 steps |
| Full steps | 79,020 |
| Planned tokens | 1,941,995,520 |
| Symbiogenesis | disabled |

The selected LR came from three matched 384-step, 9,437,184-token arms. Selection used the lowest mean held-out
loss over the final three aligned evaluations:

| Peak LR | Final-three validation mean | Final validation loss | Verdict |
|---:|---:|---:|---|
| 0.001 | 6.0457255443 | 5.8334152699 | not selected |
| 0.002 | **6.0101444324** | **5.7821794748** | selected |
| 0.003 | 6.3045441707 | 6.0585169792 | not selected |

The selection record is:

```text
/mnt/donto-data/alpha-runs/alpha-foundation-lr-pilot-20260803/selection.json
SHA-256 fcb287bc7deab8542241dd281ee4beeea22aca886f383479e80406a95ebfb35d
```

### 7.1 Frozen data identities

```text
train shard used by physical sweeps
  /mnt/donto-data/alpha-corpora/pretrain-text/pretrain-000.txt
  d993342b0bb55198c520f1f761bb0aad2812b2d8fb9c6347b4e6f9d622794d9c

held-out validation
  /mnt/donto-data/alpha-corpora/pretrain-text/foundation-val-005-64m.txt
  17e30fa2e50e1a1f116cceed95381b76edd1be595d402c4dd053bd55a7eafd60

tokenizer
  /mnt/donto-data/alpha-corpora/tokenizers/bpe-byte-12k-20260722.json
  c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24

four-shard foundation manifest
  /mnt/donto-data/alpha-corpora/pretrain-text/foundation-2b-manifest.json
  be6975e2ffe327beafdc35174321c79a778b3ac33e248eba28ab591081dcb2e0
```

The packed train cache contains 2,058,181,632 verified tokens. The planned run consumes 1,941,995,520. The
held-out slice comes from excluded `pretrain-005.txt`; none of its bytes may enter training or synthetic chat
generation.

### 7.2 Old base checkpoint used in cooperative debugging

This checkpoint is **not** the new 97M foundation candidate. It is the recoverable old 57.7M base checkpoint
used to diagnose the cooperative-kernel corruption:

```text
/mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/checkpoints/base-pretrain-step-61036.alph
SHA-256 08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049
```

An exact streaming scan found 57,688,576 finite parameters, maximum absolute value 4.641785621643066, and zero
values beyond the f16 finite range. Stored-weight overflow therefore does not explain the old cooperative B4
failure.

---

## 8. Selected Helios performance state

### 8.1 End-to-end rate

The selected production recipe processes 24 x 1,024 = 24,576 training tokens per optimizer step.

| Warm-window statistic | Tokens/s | Seconds/step |
|---|---:|---:|
| Minimum | 6,940.0 | 3.5412 |
| p10 | 7,052.7 | 3.4846 |
| Median | **7,253.8** | **3.3880** |
| p90 | 7,360.2 | 3.3390 |
| Maximum | 7,367.8 | 3.3356 |

Historical median was 3,579 tokens/s. The selected path is approximately 102.7% faster end to end. At the
selected median, the frozen token contract is about 74.37 uninterrupted GPU-hours before validation,
checkpointing, launch, and failure overhead. At the historical RTX 4090 price of USD 0.69/hour, that is about
USD 51.31 before overhead. These are estimates, not a completed run.

### 8.2 Timestamped kernel attribution

The exact profile is the mean of two selected RTX 4090 timestamp samples:

```text
1,654,743.8 us
1,686,470.3 us
mean 1,670,607.05 us
```

Timestamping changes scheduling. The 1.671-second diagnostic dispatch total must not be subtracted from the
3.388-second production median and called host overhead.

| Family | Profiled time | Share |
|---|---:|---:|
| Generic GEMM | 971.2 ms | 58.1% |
| Flash attention | 351.0 ms | 21.0% |
| Elementwise, rotary, in-place | 132.0 ms | 7.9% |
| Layout and copy | 106.6 ms | 6.4% |
| Column reduction and normalization | 78.6 ms | 4.7% |
| Other measured kernels | 31.3 ms | 1.9% |
| **Total** | **1,670.6 ms** | **100.0%** |

| Rank | Kernel | Calls | Total | Mean/call | Share |
|---:|---|---:|---:|---:|---:|
| 1 | `matmul_transposed_R42C` | 91 | 452.6 ms | 4.974 ms | 27.1% |
| 2 | `matmul_transposed_a_R42C` | 91 | 292.1 ms | 3.210 ms | 17.5% |
| 3 | `flash_attn_bwd_dkv_32_32_64` | 18 | 265.6 ms | 14.755 ms | 15.9% |
| 4 | `matmul_R42` | 91 | 226.5 ms | 2.489 ms | 13.6% |
| 5 | `column_sum` | 37 | 60.1 ms | 1.624 ms | 3.6% |
| 6 | `transpose` | 144 | 54.7 ms | 0.380 ms | 3.3% |
| 7 | `flash_attn_fwd_32_16_64` | 18 | 49.8 ms | 2.767 ms | 3.0% |
| 8 | `flash_attn_bwd_dq_32_16_64` | 18 | 35.6 ms | 1.977 ms | 2.1% |

Generate tables from raw logs rather than hand-maintaining categories:

```bash
npm run perf:profile:summary -- LOG1 LOG2 > summary.md
npm run perf:profile:summary -- --format json LOG1 LOG2 > summary.json
```

The tool parses every `[gpu_ops]` line, averages dynamic profiler kinds and physical kernel identities, reports
mean/call and dispatch share, and binds every input log by SHA-256.

### 8.3 Selected kernel environment

```bash
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_FLASH_FWD_PREFER_COOP2=0
export HELIOS_WG_SIZE=64
export HELIOS_MATMUL_REG4X2=1
export HELIOS_MATMUL_REG4X2_TRANSPOSED_B=1
export HELIOS_MATMUL_TRANSPOSED_B_COALESCED=1
export HELIOS_MATMUL_TRANSPOSED_A_COALESCED=1
export HELIOS_MATMUL_REG2X2=1
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
```

The selected physical portfolio is:

| Operation | Kernel |
|---|---|
| Ordinary GEMM | R4x2 (`R42`) |
| Transposed-B GEMM | coalesced R4x2 (`R42C`) |
| Transposed-A GEMM | coalesced R4x2 (`R42C-A`) |
| Gradient-buffer aliases | last-consumer ownership forwarding |
| Cooperative matrices | disabled |
| Symbiogenesis | disabled for this foundation contract |
| Training dtype | FP32 |

---

## 9. What actually improved Helios

| Change | Correctness evidence | End-to-end result | State |
|---|---|---:|---|
| First portable register-blocked GEMM | matched training trajectory | 3,579 -> 4,513 tok/s | superseded by later portfolio |
| Gradient ownership forwarding | exact same-source trace control; terminal validation matched | 4,121.0 -> 6,123.2 tok/s, +48.6% in diagnostic comparison | selected |
| Per-layout R42/R2 portfolio | max loss diff `9.537e-7`; max grad diff `4.308e-8`; 29 files/283 tests | 6,567.7 -> 6,836.8, +4.10% | selected |
| Coalesced transposed-B R42C | max loss diff `9.537e-7`; max grad diff `3.681e-8`; full physical suite | 6,836.8 -> 7,048.9, +3.10% | selected |
| Coalesced transposed-A R42C-A | exact loss; max grad drift `2.154e-8`; full physical suite | 7,085.0 -> 7,253.8, +2.38% | selected |

The gradient-ownership result corrected an important profiler misunderstanding. The 637
`scale_vec4x2` calls were largely autograd gradient copies, not useful arithmetic. The tape now moves the final
consumer's buffer and clones only a genuine alias. Keep the same-source control
`ALPHA_DISABLE_GRADIENT_BUFFER_MOVE=1` available for regression testing.

---

## 10. Open, experimental, and rejected Helios work

### 10.1 The next physical discriminator: row-parallel `column_sum`

The selected kernel assigns one thread to each of roughly 512 columns; each thread walks all 24,576 rows. The
candidate uses 32 columns x 4/8/16 row lanes, coalesced reads, shared-memory reduction, no atomics, and no fixed
subgroup-size assumption.

| Property | Selected | Candidate |
|---|---:|---:|
| Columns/workgroup | runtime WG width | 32 |
| Row lanes/column | 1 | 4 / 8 / 16 |
| Useful threads at width 512 | 512 | 2,048 / 4,096 / 8,192 |
| Rows/thread at 24,576 rows | 24,576 | 6,144 / 3,072 / 1,536 |
| Scratch | 0 | 512 B / 1 KiB / 2 KiB |
| Atomics | no | no |
| Subgroup assumption | no | no |
| Physical RTX time | 59.8–60.1 ms selected | open |

Local Vulkan-on-llvmpipe execution used subgroup size 8. On an awkward 257 x 96 RMSNorm weight-gradient case:

| Row lanes | Maximum absolute error |
|---:|---:|
| 4 | `2.8610e-6` |
| 8 | `4.2915e-6` |
| 16 | `4.2915e-6` |

The current local package result is 233 passed, 55 physical-GPU-gated, 0 failed. This is portability and
correctness preflight, not a speed result.

Enable a candidate only with:

```bash
HELIOS_COLUMN_SUM_ROW_LANES=4
HELIOS_COLUMN_SUM_ROW_LANES=8
HELIOS_COLUMN_SUM_ROW_LANES=16
```

Zero/default retains the selected kernel. Compatibility value `1` maps to 8 lanes.

### 10.2 Exact physical sweep

The current head provides `scripts/run_helios_column_sum_sweep.sh` and npm alias
`perf:sweep:column-sum`. It runs:

- timestamp profiles in mirrored order `control/4/8/16/16/8/4/control`;
- 20-step sustained trajectories in the same mirrored order;
- one immutable directory per row;
- exact input, source-file, runtime, source-commit, dirty-status, and dirty-patch capture;
- checkpoint disabling;
- Markdown and JSON profile tables for every timestamp row;
- a final hash manifest checked before success.

On the physical host:

```bash
cd /workspace/alpha2
npm ci
npm run build

export TRAIN_DATA=/workspace/inputs/pretrain-000.txt
export VAL_DATA=/workspace/inputs/foundation-val-005-64m.txt
export TOKENIZER=/workspace/inputs/bpe-byte-12k-20260722.json
export OUT_ROOT=/workspace/evidence/alpha-helios-column-sum-row-lanes-physical-$(date -u +%Y%m%dT%H%M%SZ)

npm run perf:sweep:column-sum
```

Paths on the pod may differ. Copy the three exact host inputs and verify their hashes against section 7 before
launch. `OUT_ROOT` must not already exist. Preserve the output onto the mounted host research tree before pod
deletion.

Promotion requires all of the following:

1. every row exits zero;
2. source/input/runtime hashes and controlled environment are complete;
3. the candidate's physical kernel identity is present;
4. losses, validation, learning rate, clipping, gradients, and outputs remain within the declared comparison;
5. the winning lane beats mirrored controls in exact timestamps and sustained end-to-end rate;
6. the complete physical GPU suite passes;
7. the result is reproduced from the selected source;
8. raw controls, failed rows, and candidate artifacts are mounted and checksum-verified.

If none wins, keep the default and record the negative result. Do not choose the least bad candidate merely
because the sweep was expensive.

### 10.3 K32 transposed-B candidate

`HELIOS_MATMUL_TRANSPOSED_B_REDUCTION_TILE_32=1` doubles K-reduction tile depth from 16 to 32 and shared memory
from 4 KiB to 8 KiB. It passed awkward local Vulkan correctness with transposed-B max error `3.338e-6`. It has no
physical speed claim and remains unselected. The current bottleneck order now places the row-parallel reduction
discriminator first because it has a complete sweep and a clear occupancy hypothesis; a next agent may still
physically compare K16/K32 if the evidence says it is the better use of a pod.

### 10.4 Cooperative matrix failure

Cooperative matrices remain disabled. The old B4 forward-only row dispatched 81 production
`transposed-B s2x2 r4x4 km4` kernels and changed first loss from `2.7419` to `6.9667` before backward.

New discriminators now include:

- production-shape rank-one ordinary/transposed-B/transposed-A gates;
- a dense, non-low-rank `1024 x 512 @ 1408 x 512^T` transposed-B oracle;
- exact f16/f32-representable inputs and zero-tolerance comparison;
- telemetry that requires direct `s2x2_r4x4` production dispatch;
- exact source commit, source hashes, controlled environment, and dirty patch capture.

Physical execution of the dense production oracle is still open. Do not infer the root cause until it runs. The
old B4 sweep lacked exact source/dirty-patch binding and cannot support a kernel-level conclusion by itself.

### 10.5 Rejected changes that must stay rejected unless the mechanism changes

| Candidate | Measured result | Verdict |
|---|---:|---|
| Four-query flash dKV unroll | dKV about 74.7% slower; dispatch graph 15.6% slower | rejected |
| `column_sum` vec4 traversal | 64.5687 ms vs 59.6313 ms, about 8.3% slower | rejected |
| Uncorrected cooperative production B4 | corrupt first forward loss | disabled |
| Batch 32 foundation shape | allocator exhaustion before first step | rejected for current shape |
| 136.9M candidate | 2,613.1 tok/s, about 8.86 days / USD 146.70 before later stages | rejected for current envelope |

Do not delete these failures. They prevent repeated dead ends and constrain future hypotheses.

---

## 11. Evidence and research map

All research, benchmark output, profiler logs, screenshots, reports, and recovered remote evidence belong on the
mounted drive under `/mnt/donto-data/donto-resources/`, never solely in `/tmp`, a pod root filesystem, terminal
scrollback, or an uncommitted worktree.

### 11.1 Canonical Helios research

```text
/mnt/donto-data/donto-resources/research/alpha-helios/
  ARTIFACTS.sha256
  CURRENT-BOTTLENECK-LEDGER-2026-08-03.md
  EVIDENCE-REGISTER.md
  MISSING-ARTIFACT-NOTICE-2026-08-03.md
  PERFORMANCE-PRIOR-ART-AND-OPPORTUNITY-AUDIT-2026-08-03.md
  PRESERVATION-POLICY.md
  PREVIOUS-ARTIFACTS-SNAPSHOT-20260803.txt
```

`MISSING-ARTIFACT-NOTICE-2026-08-03.md` records a bounded search for a historical README that could not be
found. Do not fabricate or silently replace it.

### 11.2 Canonical benchmark directories

```text
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-chat-throughput-20260802/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-gradient-ownership-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-r42-portfolio-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-transposed-coalesced-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-transposed-a-coalesced-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-column-sum-vec4-rejected-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-flash-dkv-v2-rejected-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-r42ck32-local-preflight-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-coop-production-oracle-preflight-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-coop-forward-contract-audit-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-column-sum-row-lanes-preflight-20260803/
/mnt/donto-data/donto-resources/benchmarks/alpha-helios-profile-summary-tool-20260803/
```

There is also verified shutdown/recovery evidence for the deleted RTX 4090 pod:

```text
/mnt/donto-data/donto-resources/benchmarks/alpha-runpod-shutdown-wtupxv15debnvh-20260803/
```

### 11.3 Preservation procedure

For every physical experiment:

1. capture exact source commit and full dirty status/patch before launch;
2. hash code, tokenizer, corpus, validation, configuration, and launch scripts;
3. capture device, driver, Vulkan extensions, runtime versions, controlled environment, and exact command;
4. preserve raw controls and candidates, including crashes and rejected rows;
5. record machine-readable metrics plus readable summaries;
6. copy the complete directory to the mounted evidence tree;
7. create `ARTIFACTS.sha256` and run `sha256sum -c`;
8. only then delete the exact pod.

The operating lesson from prior remote work is: “launched” is not “working.” Poll real metric deltas, process
RSS, GPU use, logs, and output growth. A PID or self-reported rate alone is insufficient.

---

## 12. Testing and correctness boundary

The last code-bearing local Helios suite result was:

```text
233 passed
55 physical-GPU-gated
0 failed
```

The last selected physical suite passed:

```text
29 files
283 tests
0 failed
```

The final script-only commit was checked with `bash -n` and negative-path checks for missing inputs and an
existing output directory. Re-run relevant tests after any code change:

```bash
cd /mnt/donto-data/workspace/alpha2
npm ci
npm run build
npm run typecheck
npm test -w @alpha/tests
bash -n scripts/run_helios_column_sum_sweep.sh
```

Use `@alpha/tests`, not `@alpha/helios`: the Helios workspace intentionally has no directly discoverable test
files, while the repository's parity, autograd, GPU, and kernel tests live in the aggregate test workspace. The
root Turbo `npm test` is also pre-existingly unsuitable because it invokes Vitest in intentionally empty
packages.

For kernel promotion, local llvmpipe correctness is necessary but insufficient. Run the complete NVIDIA physical
suite and an exact matched training trajectory. Preserve every result.

Printed loss equality alone is also insufficient. Compare at least:

- train loss at every aligned step;
- held-out validation loss;
- gradient norm and clipping coefficient;
- learning-rate schedule;
- finiteness and allocator overflow;
- kernel telemetry proving the candidate actually dispatched;
- generated output or a checksum of deterministic output where relevant;
- end-to-end tokens/s after warmup;
- exact profiled kernel time.

Repeated-token embedding-gradient scatter uses legal atomic ordering and can differ at one ULP. Bounded replay
tests now use fixed-order gather, while real training retains the fast scatter. Preserve this distinction rather
than hiding it behind a broad tolerance.

---

## 13. Recommended next-action order

### Phase A — orient and verify

1. Read this document.
2. Read the mounted `CURRENT-BOTTLENECK-LEDGER`, `EVIDENCE-REGISTER`, and preservation policy.
3. Run the Git, disk, memory, RunPod, public HF, CPU health, and BLAH metadata checks from sections 4 and 11.
4. Verify the three sweep input hashes from section 7.
5. Confirm no uncommitted work or new remote commit appeared after `b72d038`.

### Phase B — finish the row-parallel discriminator

1. Review `packages/helios/src/kernels/reduction.ts`, the new tests, and the sweep script.
2. Run local build/typecheck/Helios tests.
3. Provision a physical GPU only when the exact inputs, built source, and immutable output plan are ready.
4. Run the full mirrored profile and sustained sweep without editing rows mid-run.
5. Verify that real progress advances and the machine is healthy.
6. Analyze dynamically from `[gpu_ops]` logs and aligned trajectory metrics.
7. Promote one lane only if all correctness and performance gates pass.
8. Otherwise record the negative result and retain the selected default.
9. Copy, hash-check, and document everything before deleting the pod.

### Phase C — continue high-value performance research

Use the measured ranking rather than novelty theater:

1. transposed-B K16/K32 or a corrected reduced-precision/cooperative discriminator;
2. flash attention dK/dV work-partition redesign, not another unroll of the rejected ownership pattern;
3. GEMM epilogue fusion where a real downstream consumer removes a whole tensor pass;
4. transpose and QKV layout elimination;
5. operation-graph quotienting and safe buffer/lifetime reductions;
6. physical AMD Vulkan proof, then backend-neutral HIP/ROCm lowering where rentals lack Vulkan.

The strongest honest novel direction is a correctness-gated optimizer that chooses operation-graph
representatives using live shape/layout/device evidence while retaining portable fallbacks. Novelty is not a
substitute for a measured gain.

### Phase D — freeze the engine and train the foundation

Start the long run only after the engine recipe and physical device are selected. Use
`scripts/run_foundation_candidate_full.sh`; do not reconstruct the command by hand. Verify:

- source and dirty patch;
- four-shard manifest and packed caches;
- held-out validation hash;
- tokenizer hash;
- LR-selection report;
- selected environment;
- checkpoint cadence and resumability;
- billing guard/finalizer;
- mounted recovery destination.

During the run, measure actual metric-row progress over wall-clock windows, GPU utilization, RSS, disk growth,
allocator overflow, validation cadence, and checkpoint integrity. Compress only after raw and compressed hashes
are recorded and decompression reproduces the raw hash.

### Phase E — conversational post-training and selection

Foundation loss does not prove chat quality. Build an equal-token, behaviorally evaluated post-training program
covering:

- response initiation and answer-and-stop cases;
- short, medium, and long responses;
- natural ordinary conversation;
- instruction following and semantic contingency;
- multi-turn common ground and locally established meanings;
- counterexamples and localized revision;
- evidence-conditioned conversation and source attribution;
- questions that should be answered directly versus those requiring clarification;
- adversarial repetition, role leakage, template mismatch, and truncation;
- natural linguistic, ontological, philosophical, and pragmatic discussion.

Use strong teachers to generate candidate data, cheaper suitable models for surface variation, automated
executable checks where sound, and human authority for ambiguity, philosophy, counterexample validity, cultural
language, and final conversation preference. Preserve rejected generations and generation/review lineage.

Select by free generation on frozen whole-family and ordinary-chat holdouts. Inspect raw outputs. Record whether
failures are refusal, error, degeneration, truncation, prompt-template mismatch, or knowledge gap; those demand
different fixes.

### Phase F — publish only a behavioral winner

When a candidate beats the public model:

1. freeze the exact native checkpoint and tokenizer;
2. export Safetensors and verify golden-logit parity;
3. publish the native restart checkpoint with optimizer/RNG state;
4. create a new HF revision/model version and preserve the old one;
5. deploy a versioned runtime;
6. create a new BLAH model record rather than mutating an evaluated entry;
7. run BLAH evals and read every low-score raw log plus judge reasoning;
8. update the public Space only after the versioned backend is verified;
9. post representative before/after model outputs to Discord with why they improved.

---

## 14. Discord communication policy

The webhook is stored only in `.env.discord.local`, mode `0600`. Never print its URL. The current policy is to
report:

- a correctness-gated, reproducible speed improvement;
- a strategy-changing negative result;
- a newly isolated serious correctness bug;
- a real model-behavior improvement, with exact example input and before/after outputs;
- a major run milestone that changes cost, time, or risk.

Avoid routine noise. A useful performance post contains:

```text
what changed
exact source commit
device and model shape
before and after tok/s
profiled kernel before and after
loss/gradient/validation parity
where the mounted evidence lives
why the result changes the next decision
```

A useful model post contains only actual model text and explains why it is better. Never clean up, paraphrase, or
silently complete a model sample.

---

## 15. AMD and portability direction

The selected R42/R42C/R42C-A kernels use portable scalar FP32 Vulkan compute and have passed Mesa llvmpipe
smokes. That is not proof on physical AMD hardware. The current RunPod catalog observed by this account offered
NVIDIA only.

The intended compatibility sequence is:

1. keep backend semantics and correctness tests architecture-neutral;
2. test the native Vulkan path on a physical Radeon device when one is available;
3. inventory subgroup, shared-memory, timestamp, f16/bf16, and cooperative-matrix capabilities;
4. keep scalar portable fallbacks;
5. add a backend-neutral lowering and HIP/ROCm implementation for Instinct rental environments that do not
   expose production Vulkan;
6. compare end-to-end cost per verified training token, not just hourly rental price.

Do not claim AMD support from llvmpipe, shader compilation, or vendor-neutral source alone.

---

## 16. Jacobian Lens and model-internals track

The operator supplied a complete BLAH Lens Bundle v1 contract. It remains a required downstream deliverable for
the selected checkpoint, but no `dist/blah-lens/` directory or validated bundle exists at this handoff.

This ordering is deliberate. A Jacobian lens is checkpoint-specific. Fitting it before the behavioral winner is
immutable would spend compute on a checkpoint likely to be replaced and would invalidate its fingerprint after
the next model update.

When the winning model is frozen, the lens work must produce the complete `blah-jacobian-lens` format-v1 bundle,
including affine centering, declared estimator kind, exact final-decode parity, tokenizer parity, native VJP
finite-difference validation, matrix-orientation test, split-half convergence, byte-exact token handling,
golden fixtures, fingerprint enforcement, runtime conformance, and safe Safetensors artifacts. Do not describe
the lens as an SAE or as evidence for architectural global-workspace claims.

Current status:

```text
native lens adapter       not implemented
fit command               not implemented
dist/blah-lens            absent
validated transports      absent
public lens runtime       absent
```

The platform guide and validator were specified by the operator. Recheck the live schema and CLI before
implementation because they are external and can change.

---

## 17. Synthetic-data side program, when resumed

The eventual synthetic curriculum should be a reusable public scientific corpus rather than disposable JSONL.
The user expects Codex 5.x workers and occasional Claude agents to populate a richly normalized SQLite database,
with smarter counsel orchestrating cheap generation and batch review.

Retain at minimum:

- concept/family identity and revisions;
- natural dialogue turns separated from injected chat delimiters;
- linguistic, pragmatic, ontological, mereological, teleological, epistemic, temporal, causal, discourse, and
  conversational phenomena;
- positive, hard-negative, ambiguous, counterexample, repair, paraphrase, scope, perspective, purpose, and
  cross-domain relations;
- source fragments, licenses, model/prompt revisions, seed, raw output, parent candidates, and reviewer history;
- accepted, rejected, disputed, superseded, and retired candidates;
- exact render profiles, tokenizer versions, loss masks, releases, splits, and model exposure;
- whole-family holdouts and relation-corruption controls;
- human judgments where model reviewers are not authoritative.

Track everything cheaply in SQLite; materialize and normalize in phases. The reason not to build every table
before the first causal pilot is schedule risk, not an objection to complete provenance. Preserve raw immutable
records from day one so richer tables can be derived later.

This side program remains paused unless the operator explicitly returns to it.

---

## 18. Failure modes the next agent should avoid

- Do not claim the full foundation exists. It does not.
- Do not claim Helios speed work improved current model behavior. It did not.
- Do not restart blind continuation SFT on the old model.
- Do not select by validation loss, nonempty rate, one cherry-picked answer, or BLAH aggregate alone.
- Do not compare quality on prompts where the baseline was ineligible or failed structurally.
- Do not hide empty EOS, loops, truncation, judge errors, or prompt-template mismatches.
- Do not enable cooperative matrices in production until the exact forward corruption is resolved physically.
- Do not promote row-lane or K32 candidates from llvmpipe correctness.
- Do not repeat the rejected flash dKV unroll or vec4 reduction without a genuinely different mechanism.
- Do not treat timestamped profile time as ordinary step time.
- Do not hand-maintain brittle kernel/category maps when profiler identities can be parsed dynamically.
- Do not run a paid pod before source, inputs, outputs, and recovery paths are ready.
- Do not delete a pod before mounted evidence and hashes are verified.
- Do not leave a pod stopped or idle under the assumption that billing ended.
- Do not discard controls, failed outputs, or negative experiments.
- Do not put research artifacts on the small root disk when the mounted research tree exists.
- Do not expose credentials in documentation, shell history output, Discord, BLAH metadata, or HF repositories.
- Do not overwrite an evaluated BLAH model record with new weights.
- Do not let AlphaCorpus, Donto, or lens work replace the chatty-model product goal.

---

## 19. Completion definitions

### 19.1 Helios performance campaign

It is not complete because one kernel is faster. It is complete enough to freeze the engine when:

- the selected recipe is numerically and trajectory validated on the physical target;
- dominant costs have been reprofiled after every major gain;
- remaining candidates have documented value/risk and do not justify delaying model training;
- source and evidence are committed, pushed, mounted, and checksum-verified;
- throughput, estimated time, memory, and cost are reported from production runs;
- the pod is removed or is actively running the accepted bounded job.

### 19.2 Foundation stage

It is complete when all 79,020 planned steps are finite and auditable, held-out loss follows the contract,
allocator overflow is zero, checkpoints are recoverable, corpus/tokenizer/source identities match, and the final
native state is preserved locally and remotely. It is still not a chat model at that point.

### 19.3 Alpha product stage

The project is complete only when a versioned checkpoint is demonstrably better than the current public model in
ordinary, untouched, multi-turn free conversation; starts and stops reliably; avoids degeneration; follows
instructions; understands language and intent; preserves conversational state; and remains useful under human
blinded comparison. It must be reproducibly published with native recovery state, honest model card, working
runtime, versioned BLAH record, and exact evidence.

---

## 20. First commands for the next agent

```bash
cd /mnt/donto-data/workspace/alpha2

git status --short
git branch --show-current
git rev-parse HEAD
git log -12 --oneline --decorate

runpodctl pod list
df -h /mnt/donto-data
du -sh .
free -h

sha256sum \
  /mnt/donto-data/alpha-corpora/pretrain-text/pretrain-000.txt \
  /mnt/donto-data/alpha-corpora/pretrain-text/foundation-val-005-64m.txt \
  /mnt/donto-data/alpha-corpora/tokenizers/bpe-byte-12k-20260722.json \
  /mnt/donto-data/alpha-runs/alpha-foundation-lr-pilot-20260803/selection.json

sha256sum -c /mnt/donto-data/donto-resources/research/alpha-helios/ARTIFACTS.sha256

npm ci
npm run build
npm run typecheck
npm test -w @alpha/tests
bash -n scripts/run_helios_column_sum_sweep.sh

sed -n '1,260p' \
  /mnt/donto-data/donto-resources/research/alpha-helios/CURRENT-BOTTLENECK-LEDGER-2026-08-03.md
```

Then decide from current evidence whether to run the physical row-lane sweep, not from the excitement of having a
new agent session.

---

## 21. Final handoff statement

At transfer, all known local changes are committed and pushed. No paid Alpha pod is running. The public model is
healthy as a service but still honestly marked `quality_gate=FAIL`. The full foundation run has not started. The
selected Helios recipe has a measured 7,253.8-token/s median on the exact foundation shape. The next candidate,
row-parallel `column_sum`, is locally correct and fully scripted for a mirrored physical comparison but has no
physical result. Canonical research and benchmark evidence is on the mounted drive with preservation rules and
checksums. The immediate job is to verify the handoff, run or reject that discriminator cleanly, continue only
high-value measured optimization, then freeze the engine and train the foundation that can support a genuinely
better conversational model.
