---
name: alpha-perf
description: Boot and drive the alpha2 native+Vulkan GPU-performance autoresearch loop — scaling the from-scratch Helios stack toward 50,000 tokens/second for a ~100M-parameter model on an RTX 3070. Provides the pod access, the goal gate, the alphaperf sqlite research DB (measurements + the 2,644-operation universe + the experiment loop), the measurement discipline, the SASS capture recipe, the current state and the proven/refuted levers. Use whenever working on alpha2/Helios throughput, the native or Vulkan backend, the tensor-core GEMM, the sm_86 assembler, kernel lowering, or any "make it faster" task on this stack.
---

# alpha-perf — the 3070 → 50k tok/s autoresearch loop

You are a GPU-performance research agent on **alpha2** (`/mnt/donto-data/workspace/alpha2`),
a from-scratch CUDA-free training stack. Two backends run the same model: **native**
(our own sm_86 driver + SASS assembler, no CUDA, no cuBLAS) and **Vulkan**. The job is
to push both toward groundbreaking throughput for a ~100M-parameter GPT on **one RTX
3070** — the only card the sm_86 emitter targets.

**THE GOAL:** native **50,000 tok/s** and Vulkan **10,000 tok/s**, at ~100M params, on a
3070, at the same commit, loss preserved. (The gate's historical target is 30k; the
standing north star is 50k. Vulkan already passes ~12k. Native is the frontier: ~19,900
as of 2026-08-06, up from 1,179.)

This is not a checklist — it is a **loop**. Profile, find the one binding constraint,
consult the operation universe for the relevant primitive and its lowering, build it
*properly* with tests, measure it in isolation and end-to-end, record the result (win or
loss) in the DB, and repeat. The DB is the loop's memory; keep it current every cycle.

---

## 0. The three non-negotiable laws of this stack

1. **NEVER TAKE A SHORTCUT. Build everything perfectly regardless of complexity.** No
   4-byte-when-128-is-right, no "good enough" tolerance, no leaving the hard case for
   later. Correctness-first-then-optimize is proper; shipping the shortcut is not.
2. **MEASURE — do not reason about performance.** Every instrument on this stack has
   lied at least once (see §4). A number without a paired before/after at a known commit
   is a guess. A "faster" claim without a measurement is a bug waiting to be found.
3. **LOSS IS BIT-IDENTICAL unless a change is explicitly a numerics change.** Every
   throughput commit this program has made kept `loss 9.5818` to the digit. A moved loss
   is a correctness regression until proven otherwise.

Corollary — **a dead lever stays dead.** The DB records REFUTED experiments precisely so
the same idea is not re-tried. Read `alphaperf.py loop` before proposing anything.

---

## 1. Boot the machine (one minute)

The GPU is on a RunPod pod; the git history is on the box. Source moves box → pod.

```bash
apod                       # ssh into the pod (interactive)
apod '<cmd>'               # run a command on the pod
apod-sync                  # rsync the working tree box → pod (excludes .git, node_modules, *.node)
```

Build + gate, always in this order:

```bash
apod-sync
apod 'cd /workspace/alpha2 && node packages/helios/native/build-stack.mjs'   # native stack + its C tests
apod 'cd /workspace/alpha2 && node packages/helios/native/build.mjs'          # the Vulkan addon
apod 'cd /workspace/alpha2 && bash scripts/goal-gate.sh'                       # BOTH backends, prints PASS/FAIL
```

- **`HELIOS_VIDMEM=1` is mandatory** — without it a 105M model measures 31 tok/s instead
  of ~2,882 (38x), because its tensors sit in host memory. `goal-gate.sh` sets it.
- Native toolchain lives under `/home/ajaxdavis` (NOT `ajax`); build with the env prefix
  in §7 or just use `build-stack.mjs` which handles it.
- **Two addons, two build scripts:** `build-stack.mjs` builds the NATIVE one, `build.mjs`
  builds the VULKAN one. A failed build leaves the previous `.node` in place, so the tests
  keep running the OLD binary — read the BUILD output, not only the test output.
- The card idles at 210 MHz vs 2100 and cannot be clock-locked in the container, so
  end-to-end tok/s has a ~1-2% run-to-run spread. Warm by TIME, not iteration count.

---

## 2. The autoresearch loop (the actual method)

```
  ┌─ profile the full step → find the ONE binding constraint ────────────────┐
  │  packages/tests/profile-gpu-by-op.mjs (drained, per-op share)            │
  │  packages/tests/micro-*-bandwidth.mjs (isolated, honest us/GB/s)         │
  │  probe-gemm-rate.mjs (GEMM TFLOP/s by shape+layout, L2-evicted)          │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
  ┌─ consult the operation universe: which primitive/op, what lowering ──────┐
  │  alphaperf.py sql "SELECT * FROM operation WHERE family='gemm' ..."      │
  │  gpu-op-universe/docs/{LOWERING_GUIDE,MATRIX_MULTIPLICATION_UNIVERSE}.md │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
  ┌─ build it PROPERLY, with tests at every layer ──────────────────────────┐
  │  missing ISA? → capture (§5) → encode → ISA test → hardware test        │
  │  new kernel? → correctness vs a reference FIRST, then optimize          │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
  ┌─ MEASURE: isolated micro AND end-to-end goal-gate ──────────────────────┐
  │  record BOTH; a kernel win that does not move the step is not a win     │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
  ┌─ record in alphaperf: gate/kernel/gemm/op/experiment/finding ───────────┐
  │  advance the op's impl_status; log the experiment (confirmed OR refuted) │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 ▼  commit (loss-checked), then repeat
```

Start every session by reading the state:

```bash
python3 tools/alphaperf.py loop      # throughput, recent experiments, open + refuted levers
python3 tools/alphaperf.py roadmap   # the operation frontier: what's built, what's stub
```

---

## 3. alphaperf — the research database (`tools/alphaperf.py`, DB at `alphaperf.db`)

The single home for measurements, the operation registry, and the experiment loop. It is
gitignored (a workbench) and rebuilt deterministically by `tools/alphaperf_backfill.sh`.
**After any schema or seed change, rebuild:** `rm -f alphaperf.db* && python3
tools/alphaperf.py init && bash tools/alphaperf_backfill.sh`.

Tables:

| table       | what it holds |
|-------------|---------------|
| `gate`      | end-to-end tok/s per backend per commit (the headline number) |
| `kernel`    | isolated single-kernel microbenchmarks (us/call, GB/s vs a control) |
| `gemm`      | GEMM rate probes by shape + layout (nn/nt/ta), TFLOP/s |
| `commit_log`| what each commit moved (before/after tok/s) |
| `finding`   | measured findings AND refutations, status confirmed/refuted/todo/inprogress |
| `experiment`| one turn of the loop: hypothesis, lever, before/after, verdict |
| `operation` | **the 2,644-op universe, now living in the DB** + an implementation axis |
| `isa`       | the ISA coverage snapshot (encoded/captured/missing) |

Commands (every row carries the commit it was taken at):

```bash
# record a measurement
alphaperf.py gate <commit> native <tok_s> --batch 24 --loss 9.5818 --note "..."
alphaperf.py kernel <commit> layerNorm 1536x640 34.4 228.9 --pct 66 --note "..."
alphaperf.py gemm <commit> "mlp fc B^T" 1536 2560 640 nt 1 20.48 246

# the operation universe (registry lives IN the DB)
alphaperf.py op-import gpu-op-universe/catalog/operation-registry.json   # one-time seed
alphaperf.py op <op_id> <impl_status> --tflops N --roofline N --ref file:sym --commit HASH --note "..."
#   impl ladder: stub → captured → encoded → tested → measured → optimized
alphaperf.py roadmap                     # the implementation frontier

# the loop's memory
alphaperf.py experiment "<hypothesis>" --lever cp.async --before 19116 --after 19988 \
             --verdict confirmed --op <op_id> --commit HASH
alphaperf.py finding gemm "<summary>" --value "..." --status todo|inprogress|confirmed|refuted
alphaperf.py loop                        # the whole research state
alphaperf.py sql "<read query>"          # arbitrary
```

**The DB is meant to EVOLVE.** Every cycle: advance the touched op's `impl_status`, log the
`experiment` (even a loss — especially a loss), and update the driving `finding`. When the
schema needs a new column or table to track a new kind of improvement, add it to the SCHEMA
in `alphaperf.py` and the backfill — the tracker is expected to get better at tracking.

---

## 4. The measurement discipline (every one of these has bitten this stack)

- **The drained per-op profiler OVERSTATES by ~a drain per call** and charges an untracked
  op's GPU time to whichever tracked op runs next. It has misled three times. Use it for
  the SHARE, cross-check the absolute with an isolated micro-benchmark. Its coverage line
  (profiled total ÷ step spin) must exceed 1.0; below 1.0 means ops are missing.
- **Isolated micro-benchmarks are the truth:** fire N times, drain ONCE, against an
  elementwise control on the same card in the same process. `micro-norm-bandwidth.mjs` is
  the template. Release + `endStep()` inside the loop or the pool exhausts and you measure
  the ALLOCATOR (an 800x carve), not the kernel — the defect that made one GEMM layout read
  4.5x slow for a year.
- **A read of a tensor's `.data` drains the queue** (there is no `flushAndWait`); read FOUR
  BYTES (a beacon tensor), not the whole result, or the PCIe copy is in your timed region.
- **Evict L2 before a timed GEMM** (stream 32 MB through a copy) or you measure the previous
  case's cache state.
- **ABLATION is the arbiter when instruments disagree:** stub the op, re-time the step.
  Three instruments once disagreed 23x (25.1 / 2.5 / 1.1 ms); ablation was right.
- **When two instruments disagree, distrust the one saying "no problem"** — it is the one
  that ends the search.

---

## 5. Capturing a missing ISA instruction (the from-scratch stack lacks most of them)

The native assembler (`packages/helios/native/hephaestus/`) implements ~40 of sm_86's
instructions. The rest are **stubs that ABORT with this recipe** (`sm86_stub.c`), and the
coverage register (`coverage.h` + `hp_isa_coverage`) names every one with what its absence
costs. `packages/tests/audit-isa-coverage.mjs` fails if a catalogued mnemonic has no row.

To add one — the five steps every encoder here went through, no shortcuts:

1. **Write it in CUDA at least TWICE** with different registers/immediates
   (`tools/<insn>_capture.cu`) — one capture cannot tell an operand's field from a constant
   that sits in it. `tools/shfl_capture.cu` (11 kernels) is the worked example.
2. `nvcc -arch=sm_86 -cubin -o x.cubin x.cu && cuobjdump -sass x.cubin`
   (nvcc is NOT on PATH on the pod: `export PATH=/usr/local/cuda-12.8/bin:$PATH`).
3. **Derive every field from what MOVES between two captures**, never from where it looks
   like it should be. Get both 64-bit words — the HIGH word carries the control field and
   sign bits a low-word check misses.
4. **Encode it** (`sm86_mem.c`/`sm86_float.c`/`sm86_flow.c`) and **assert the exact words**
   in `test/hephaestus_isa_test.c`, including that no field reaches its neighbour.
5. **Only then use it**, flip its coverage row to ENCODED, and measure.

⚠️ **A wrong encoding does not fault** — it executes a different instruction or reads
registers nobody wrote, and returns a finite, plausible, WRONG matrix. This stack has hit
that six times. And **bit-correct is not the whole contract for async ops:** cp.async
needed the SCOREBOARD wired in the control fields (`LDGDEPBAR` must set write-barrier 0) —
the encoding matching ptxas was necessary but not sufficient. Decode a full ptxas pipeline
loop's control fields before relying on a new async instruction.

---

## 6. Where the number is, and where it's going (state as of 2026-08-06)

Read the live version with `alphaperf.py loop`; this is the shape of it.

- **The step is GEMM-bound (~69% of GPU). The non-GEMM half is AT the bandwidth roofline**
  (gelu/geluBwd/addInplace 340-402 GB/s vs a 345 control) — the reduction was the one
  off-roofline op and warp-shuffle fixed it (layerNorm 65→34 us). So the only non-GEMM
  lever left is FUSION; everything else is the GEMM.
- **The GEMM k-loop is 42 instructions, only 33% tensor** — it moves each tile through a
  register round trip (12 LDG + 8 F2FP + 8 STS). That is the whole reason it sustains
  15-21 TFLOP/s against cuBLAS's 24-32.
- **THE LEVER: cp.async + f16-in-memory staging.** When operands are f16 in global, one
  128-bit `LDGSTS` moves a tile row straight to shared — no register, no pack, no store —
  and per k-step the 12+8+8 collapse to 2 LDGSTS (42→~18 instrs, tensor 33→78%). f16 in
  memory is numerically FREE (the staged path already rounds operands to f16 before the
  tensor cores). **DONE + MEASURED:** `emit_hmma_cpasync_f16` (NT, single-buffered) is
  correct on hardware and wired to `hl.matmulCpasync`. Measured on the m1536 NT forwards
  (`probe-cpasync-gemm.mjs`): **+10%** — qkv 21.7→23.8, mlp fc 21.1→22.8, lm head
  22.7→25.8 TFLOP/s, lm head reaching cuBLAS's 24-32 range. NOT the 2x the instruction
  count predicted: single-buffered still DEPBAR-waits per k-step and the register round
  trip was only ~10% of the cost. **M2 double-buffering REFUTED (0.67x, worse than staged)**
  — occupancy already hides the copy latency across blocks (register-limited ~4 blocks/SM),
  so within-block overlap is unneeded and its extra instructions + doubled shared only cost.
  **Remaining:**
  - **M4 (next)** wire the single-buffered cp.async GEMM into the model: cast qkv/mlp-fc
    NT-forward operands to f16 buffers, route through hl.matmulCpasync, confirm loss
    bit-identical, measure the gate delta (~+3-4% end-to-end expected — the +10% is on the
    forward projections, ~30-40% of the step).
  - **f16 accumulate — RE-TEST now.** It was refuted as +4% because the kernel was
    issue-bound not tensor-bound; cp.async moved it toward tensor-bound (24 vs 21), so its
    90.3-vs-45.5 ceiling may finally pay. Measure again on the cp.async kernel.
  - **M5** the nn/ta layouts (non-K-contiguous operand needs transpose-on-stage).
- **Also open (todo in the DB):** batched Q@K^T (nt) is 2-3x slow — 4 staging requests/warp
  vs 1; fix is a 32-k staging tile (~2.1 ms).
- **REFUTED — do not retry:** f16-accumulate alone (4%, kernel is issue-bound not
  tensor-bound — it pays only AFTER cp.async makes the kernel tensor-bound), bigger batch
  (rate falls as operands leave cache), split-K (+8% time), flash-attention (~3% at seq64),
  every geometry/register-tile/occupancy/barrier sweep (~1% each).

---

## 7. File map + build incantation

```
packages/helios/native/
  hephaestus/     the sm_86 assembler — sm86_*.c (encoders), isa.h (opcodes+fields),
                  coverage.h + sm86_stub.c (the ISA coverage register), control.h
  prometheus/     kernel codegen — hmma.c (tensor-core GEMM incl. emit_hmma_cpasync_f16),
                  reduction.c + normalize.c (warp-shuffle reductions), cast.c (f32↔f16)
  aether/gaia/hermes/chronos/helios/   driver, memory, channels, timeline, facade
  tools/          *_capture.cu (ISA captures), hmma_dump.c, l2_bandwidth.cu, ...
  test/           the C hardware + ISA tests (prometheus_hw_test.c is the kernel harness)
packages/tests/   the JS instruments — micro-*, probe-*, profile-*, diff-*, bench-*
scripts/goal-gate.sh          THE gate (both backends, PASS/FAIL)
tools/alphaperf.py            the research DB CLI
tools/alphaperf_backfill.sh   deterministic DB rebuild
gpu-op-universe/              the 2,644-operation grammar (now seeded into alphaperf.db)
```

Build a native crate/binary niced on the shared box:

```bash
RUSTUP_HOME=/home/ajaxdavis/.rustup CARGO_HOME=/home/ajaxdavis/.cargo \
  PATH=/home/ajaxdavis/.cargo/bin:/usr/bin:/bin:$PATH CARGO_TARGET_DIR=/tmp/da-target \
  nice -n19 ionice -c3 cargo build -p <crate>
```

---

## 8. The rules, restated because they are the difference

- Build perfectly, no shortcuts, however complex. Correctness-first-then-optimize; never
  ship the shortcut.
- Measure everything; trust the isolated micro over the drained profiler; ablation settles
  ties.
- Loss bit-identical unless the change IS a numerics change.
- Record every cycle in `alphaperf` — wins AND losses. Advance the op's impl_status. A dead
  lever stays dead.
- Never give up on a hard kernel: a wrong matrix is a cue to dump the SASS
  (`tools/hmma_dump.c` + `nvdisasm`) and read it, not to guess a sixth time.
- The DB and the operation dataset are meant to get better at their job over time — improve
  them as the work teaches you what to track.
