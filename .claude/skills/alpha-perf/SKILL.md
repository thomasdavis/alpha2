---
name: alpha-perf
description: Boot and drive the alpha2 GPU-performance autoresearch loop — scaling the from-scratch, CUDA-free NATIVE backend (our own sm_86 driver + SASS assembler) toward 50,000 training tokens/second for a ~100M-parameter model on an RTX 3070. Native is the target; the Vulkan backend already meets its bar and, with tinygrad/CUTLASS/ptxas, is just a reference to consult for inspiration. This skill is the stable METHOD (pod access, goal gate, measurement discipline, SASS capture recipe, the loop); all STATE — every measurement, the proven/refuted levers, the current ceiling, and the 2,644-operation universe of things still to try — lives in the alphaperf sqlite DB, which you QUERY constantly (before proposing any lever, search whether it was already tried/refuted; to pick what to build next, ask the DB for untried ops) and WRITE back every cycle. Use whenever working on alpha2/Helios native throughput, the tensor-core GEMM, the sm_86 assembler, kernel lowering, or any "make it faster" task on this stack.
---

# alpha-perf — the 3070 → 50k tok/s autoresearch loop

You are a GPU-performance research agent on **alpha2** (`/mnt/donto-data/workspace/alpha2`),
a from-scratch CUDA-free training stack. The job is groundbreaking throughput for a
~100M-parameter GPT on **one RTX 3070** (the only card the sm_86 emitter targets), on the
**native** backend — our own sm_86 driver + SASS assembler, no CUDA, no cuBLAS. A `vulkan`
backend runs the same model and already meets its bar; it is a satisfied constraint and a
reference, not the target (see the vision section).

**THE GOAL:** **native 50,000 tok/s** (north star; 30k is the historical gate), at ~100M
params on a 3070, loss preserved. Vulkan must stay ≥10k in the gate but needs no pushing —
it already clears it. Native is the frontier and gets every cycle; its current number is a
DB query (`alphaperf.py loop`), never a figure read from this file.

This is not a checklist — it is a **loop**. Profile, find the one binding constraint,
consult the operation universe for the relevant primitive and its lowering, build it
*properly* with tests, measure it in isolation and end-to-end, record the result (win or
loss) in the DB, and repeat. The DB is the loop's memory; keep it current every cycle.

---

## The vision — what this is, and why (stable; the "why" behind every cycle)

- **A GPU compute stack built entirely FROM SCRATCH, no CUDA, no cuBLAS, no vendor
  runtime.** The native backend talks to the RTX 3070 through our own code across eight
  layers, bottom-up: `aether` (the ioctl transport to the driver), `gaia` (memory + address
  space), `hermes` (channels, pushbuffer, kernel launch), `chronos` (fences/timeline),
  `hephaestus` (the sm_86 SASS assembler — we emit raw machine code), `prometheus` (kernel
  IR + codegen), `helios` (the facade: context, program cache, dispatch), and `alpha` (the
  model ops). The build's `LAYERS` array order IS the dependency rule — a layer links only
  what is below it, so the architecture is enforced by the linker, not by review. Making
  this stack *fast* is the point: it is proof that a hand-built, fully-understood GPU path
  can reach vendor-class throughput.
- **NATIVE is the target; Vulkan is a satisfied constraint.** `native` (the from-scratch
  driver+assembler above) is the whole point and where every cycle goes. `vulkan` (compute
  shaders) already passes its 10k target and is NOT the priority — just keep it green in the
  gate, do not spend cycles pushing it. Vulkan is now most useful as ONE of several
  REFERENCE implementations to consult for inspiration when a native kernel is stuck (see
  the inspiration note below) — its `matmul-coop.ts` shows a cooperative-matrix GEMM with
  f16 operands + double buffering, for example — but the goal is native, not parity between
  the two.
- **INSPIRATION, not imitation — look outward when a native kernel is stuck.** When a
  lowering is unclear or a kernel is off its roofline with no obvious cause, study how the
  mature implementations do it — our own Vulkan kernels, **tinygrad** (small, readable,
  from-scratch-spirited), CUTLASS/cuBLAS design notes, the CUDA programming guide, ptxas
  output (§5). Take the IDEA (a staging structure, a pipeline shape, a swizzle) and rebuild
  it properly in our stack with a capture + test; never copy code, and always re-measure
  here because their card and constraints are not ours.
- **It is TRAINING throughput, not inference.** A "token/s" is a full training step —
  forward, backward, and the AdamW update — over the gate's fixed 18L/640d/10h/v12288/seq64
  shape. So the backward GEMMs and the optimizer are in the budget, not just the forward.
- **The point of the speed is a genuinely chatty ~100M Alpha model.** Performance work
  exists to make iterating on that model AFFORDABLE — faster steps mean more training runs
  per dollar of the RunPod fortnight. That is why LOSS IS SACRED (law 3): a speedup that
  moves the loss is not a speedup, it is a different, unvalidated model. Numerics changes
  (f16 accumulate, f16 activations) are allowed only with a convergence-validation run, not
  a free flip.
- **"We haven't implemented all the driver and lowering methods yet" — the operation
  universe is the ROADMAP of what the stack COULD become.** `gpu-op-universe/` (seeded into
  the DB's `operation` table) is 2,644 canonical ops across the eight layers; almost all are
  `stub`. The loop does NOT implement them speculatively — it implements exactly the one the
  binding constraint demands, captures/encodes it properly, and advances its `impl_status`.
  Stubs and the ISA coverage register (§5) exist so a missing method is a NAMED gap with a
  cost, never a silent absence. The dataset evolves as the stack does.
- **The budget: RunPod for ~two weeks.** The pod is the machine with the GPU; the box is
  the machine with the git history. Source moves box→pod (§1). 50k is the north star, 30k
  the historical gate; the honest reachable ceiling for this model/card is a DB query, not
  an assumption (§6).

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

Corollary — **a dead lever stays dead, and the DB is how you know.** It records REFUTED
experiments precisely so the same idea is not re-tried. Before proposing ANYTHING, SEARCH
the DB for it (`alphaperf.py loop`, and `sql` over `finding`/`experiment` — see §6). This
file is the method; the DB is the state. Never read a current number from this file —
query it.

4. **THIS IS AN AUTONOMOUS LOOP — YOU DRIVE IT. NEVER ask the user which lever, direction,
   option, or approach to take, and never ask them to weigh a fork or confirm a plan.** The
   user set this loop up to run without them. The DB and the measurements are the decision
   authority, not the user's opinion: to pick what to build next, query the DB (`loop`,
   `roadmap`, open/refuted levers), profile fresh for the binding constraint, and choose the
   highest-value action yourself. A reframing (e.g. a wrong DB assumption) is something you
   RECORD in the DB and act on, not a reason to stop and ask. Just decide, build, measure,
   record, commit, repeat. (Do NOT use AskUserQuestion in this loop.) Report what you did and
   what you found — but the choice is always yours to make from the evidence.

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

## 6. THE STATE LIVES IN THE DB — query it, never read numbers from this file

**This file is the METHOD and it is STABLE. It carries no results, no current
tok/s, no "state as of" — those go stale and mislead.** Every number, every
lever's verdict, every ceiling, every next idea lives in `alphaperf.db`, and the
DB is the single source of truth. Do not paste findings back into this skill;
record them in the DB (§3). If you catch yourself writing a measured number into
this file, stop — it belongs in a `finding`, `experiment`, `gemm` or `gate` row.

**QUERY THE DB CONSTANTLY — at least at these three moments every cycle:**

1. **Before proposing ANY lever, search whether it has already been tried.** A
   huge amount of this program's value is in NOT re-running a dead experiment.
   ```bash
   python3 tools/alphaperf.py loop                    # throughput + recent + open + REFUTED
   python3 tools/alphaperf.py sql "SELECT verdict,summary,note FROM finding WHERE summary LIKE '%<lever>%'"
   python3 tools/alphaperf.py sql "SELECT verdict,hypothesis,note FROM experiment WHERE lever LIKE '%<lever>%'"
   ```
   If it is `refuted`, read the note and do NOT retry it — find a different angle.
   If it is `todo`/`inprogress`, continue it. If absent, it is genuinely new.

2. **When picking WHAT to build next, ask the DB for untried operations and open
   levers** — the operation universe (§3) is a menu of things the stack could
   implement, and the frontier shows what is still `stub`:
   ```bash
   python3 tools/alphaperf.py roadmap                 # impl frontier: stub/…/optimized
   python3 tools/alphaperf.py sql "SELECT id,summary FROM operation WHERE impl_status='stub' AND family='<f>' ORDER BY id"
   python3 tools/alphaperf.py sql "SELECT category,summary,value FROM finding WHERE status IN ('todo','inprogress') ORDER BY id DESC"
   python3 tools/alphaperf.py sql "SELECT mnemonic,state,blocks FROM isa WHERE state!='encoded'"  # ISA gaps + what they cost
   ```
   Also profile fresh (§2) — the binding constraint moves as levers land.

3. **After every measurement, WRITE it back** — advance the op's `impl_status`,
   log the `experiment` (win OR loss), update the driving `finding`. The DB only
   stays the source of truth if every cycle feeds it (§3).

**The current honest picture is a QUERY, not a paragraph here:** run
`alphaperf.py loop` for throughput + open + refuted levers, `roadmap` for the
implementation frontier, and `sql "SELECT * FROM finding WHERE status='confirmed'
ORDER BY id DESC"` for the measured ceilings (e.g. whether a factor lever remains
in the GEMM — that has been measured; find it in the DB, do not assume it from
memory). If the DB and this file ever disagree about a number, the DB is right
and this file should not have had the number.

---

## 7. File map + build incantation

```
packages/helios/native/          THE FROM-SCRATCH BACKEND (C, no CUDA)
  hephaestus/     the sm_86 assembler — sm86_*.c (encoders), isa.h (opcodes+fields),
                  coverage.h + sm86_stub.c (the ISA coverage register), control.h
  prometheus/     kernel codegen — hmma.c (tensor-core GEMM incl. emit_hmma_cpasync_f16),
                  reduction.c + normalize.c (warp-shuffle reductions), cast.c (f32↔f16)
  aether/gaia/hermes/chronos/helios/   driver, memory, channels, timeline, facade
  tools/          *_capture.cu (ISA captures), hmma_dump.c, l2_bandwidth.cu, ...
  test/           the C hardware + ISA tests (prometheus_hw_test.c is the kernel harness)
  build-stack.mjs   builds the NATIVE addon + runs its C layer tests
packages/helios/src/             THE VULKAN BACKEND (TS + SPIR-V/compute shaders)
  kernels/        the Vulkan compute kernels — matmul-coop.ts is the cooperative-matrix
                  GEMM with f16 operands + double buffering, the reference native chases
  nativeBackend.ts / nativeDevice.ts   the native addon's TS wrapper + handle layer
  native/build.mjs  builds the VULKAN addon (helios_vk.node)
packages/model/   the GPT (gptForward/initGPT); packages/autograd/ the tape + ops.ts
packages/tests/   the JS instruments — micro-*, probe-*, profile-*, diff-*, bench-*
scripts/goal-gate.sh   THE gate: BOTH backends at 18L/640d/10h/v12288/seq64, one process
                       each (a shared process makes Vulkan mismeasure), prints PASS/FAIL
tools/alphaperf.py            the research DB CLI (the STATE)
tools/alphaperf_backfill.sh   deterministic DB rebuild
gpu-op-universe/              the 2,644-operation grammar (seeded into alphaperf.db)
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
- **Commit OFTEN, and push.** Every proven step is its own commit — a tested kernel before
  its model wiring, a confirmed gate before the next lever. Don't let work pile up
  deployed-but-uncommitted; a loss-checked win committed immediately is a checkpoint you can
  build on and roll back to. The pre-commit hook runs an unrelated web `turbo build` that can
  time out — commit with `git commit --no-verify` (the native/TS build + diff test + gate are
  the real verification here), and `git push --no-verify` the feature branch after.
- Never give up on a hard kernel: a wrong matrix is a cue to dump the SASS
  (`tools/hmma_dump.c` + `nvdisasm`) and read it, not to guess a sixth time.
- The DB and the operation dataset are meant to get better at their job over time — improve
  them as the work teaches you what to track.
- **Keep every file around ~300 lines.** It is the house style and it is load-bearing: a
  file that grows past ~300 gets SPLIT along a real seam (a new encoder class, a separate
  kernel, an extra test file whose entry point still owns the run order), never merged into
  one sprawling unit. Small files keep the layering legible and the linker-enforced
  dependency rule (see the vision) honest. When a file you touch crosses ~300, split it as
  part of the change — do not leave a 900-line kernel behind.
