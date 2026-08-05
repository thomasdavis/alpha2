# The from-scratch GPU stack

**Status:** contract of record, adopted 2026-08-04. Supersedes Vulkan as Alpha's path to the GPU.

> **2026-08-05 — the native path is now FASTER than the Vulkan one it replaces.**
> On the 2L/64d/4h benchmark: **native 2943 tok/s against Vulkan's 612, 4.5x**, and
> 2x `cpu_ref`. Loss identical to `cpu_ref` at 4.1834; `run_native_gates.sh` green,
> 40 passed / 0 skipped. It was 0.14x that morning.
>
> Almost none of it was kernels. The step was 65% **driver** (a fresh tensor
> allocation cost 802 µs against 1.0 µs from the pool, and nothing was ever freed),
> and most of the rest was the host reading **write-combined** memory — 161x slower
> than ordinary memory, paid by every broadcast, slice and concatenation. Tensors
> are now carved from slabs mapped once, and mapped **cached**. Full account, with
> the wrong turns and what killed them, in
> [`resume/X61-NATIVE-BEATS-VULKAN-2026-08-05.md`](resume/X61-NATIVE-BEATS-VULKAN-2026-08-05.md).
>
> **Known defect, deliberately left in place:** nothing frees intermediates, in the
> benchmark or in training — `trainer.ts` probes for `releaseGpuTensor` and this
> backend spells it `release`, so reclamation is silently off. The alias is one
> line and is not yet safe to add; see X61's closing section.

---

## 1. Why

Alpha's soul constraint is that **every training FLOP goes through Alpha's own code**
([`GOAL.md`](../GOAL.md) §1). That is true today of the tensor library, the autograd tape, and the
kernels. It is not true of the path to the hardware.

`packages/helios` reaches the GPU through **78 Vulkan entry points**, and the SPIR-V we hand-emit is
compiled to machine code by NVIDIA's driver. Vulkan supplies the device model, the memory model, the
queue semantics, the dispatch protocol, and the compiler. That is a large borrowed layer sitting
underneath a project whose entire premise is not borrowing.

This replaces it. We keep the vendor **kernel module** as the syscall boundary — that is what makes the
stack runnable on rented pods at all — and own everything above it, **including the machine code**.

## 2. Decisions

| | |
|---|---|
| **Depth** | Our own ISA machine code. No vendor compiler anywhere in the runtime. |
| **Boundary** | `ioctl` into the unmodified vendor kernel module. All of userspace is ours. |
| **Target** | NVIDIA **sm_86** (Ampere). RunPod offers 33 GPU types and every one is NVIDIA. |
| **Language** | **C**, for the whole GPU stack. TypeScript keeps only the `Backend` shim. |
| **Migration** | Hard replace. `packages/helios` is rewritten in place; no runtime fallback. |

Two consequences worth stating up front.

**The ISA is the crux.** NVIDIA does not document SASS. `Hephaestus` is therefore reverse-engineering
validated against `nvdisasm`, not implementation against a specification. Development runs on an **RTX
3070 at $0.13/hr** — 3070, 3080 and 3090 are all sm_86, so encodings are bit-identical to the card we
train on.

**C is a performance decision, not only a purity one.** X39 measured host work at 21.6% of a step,
rising to ~46% once cooperative arithmetic lands. A large share of that is TypeScript packing dispatch
records into an `ArrayBuffer` for the native decoder to unpack (`desc_update` 23.4%, `push_const` 8.9%,
`decode` 2.4%). Moving the stack into C deletes that marshalling: napi is crossed **once per step**
instead of once per operation.

## 3. Architecture

Scoped to the Helios GPU stack. Everything above the `Backend` interface — `tensor`, `autograd`,
`model`, `train`, `cli` — stays TypeScript and should not notice this happened.

```
  TYPESCRIPT (unchanged)
  Alpha        tensor, autograd, model, train
    |
  helios/src   thin Backend shim — forwards, marshals nothing
    |
  ============ napi: ONE crossing per step ============
    |
  C (packages/helios/native/)
  helios/      facade: Backend entry points, op graph, scheduling
    |
  prometheus/  kernel IR -> codegen            (replaces spirv.ts + helpers.ts)
    |
  hephaestus/  SASS assembler: encoding, control bits, register allocation
    |
  chronos/     fences, semaphores, timeline ordering
    |
  hermes/      channels, GPFIFO/pushbuffer, QMD kernel launch
    |
  gaia/        physical memory, GPU virtual address space, mappings
    |
  aether/      ioctl transport: device nodes, RM object model
    |
  ============ syscall ============
  nvidia.ko    (vendor, unmodified)
```

The names continue Alpha's solar/Greek line, and each one is chosen to describe the job:

| Layer | Meaning | Owns |
|---|---|---|
| **Aether** | the medium everything passes through | `/dev/nvidiactl`, `/dev/nvidia0`, `/dev/nvidia-uvm`; the RM object model (client, device, subdevice, VA space, channel group). **Every `ioctl` struct.** Nothing above it makes a raw syscall. |
| **Gaia** | the ground everything stands on | Video and system memory allocation, GPU virtual address reservation and mapping, host staging maps, buffer lifetime. Replaces `vkAllocateMemory`, `vkMapMemory`, `vkGetBufferDeviceAddress`. |
| **Hermes** | carries the work | GPFIFO ring, pushbuffer method encoding for the Ampere compute class, QMD construction, constant-bank setup for kernel parameters, doorbell. Replaces command buffers, descriptor sets, push constants, `vkCmdDispatch`. |
| **Chronos** | time | GPU-written semaphore values, fence waits, the monotonic timeline the backend already assumes. Replaces `vkQueueSubmit` signalling and `vkWaitSemaphores`. |
| **Hephaestus** | the forge | sm_86 instruction encoding, the scheduling/control bits (stall counts, yield, barrier indices, dependency masks), register allocation. The hardest thing here. |
| **Prometheus** | gives fire to the kernels | A small typed IR and codegen, so kernels express intent rather than bit patterns. Direct replacement for `spirv.ts` + `kernels/helpers.ts`. |
| **Helios** | the sun; unchanged name | The `Backend` facade, op graph and scheduling. |

### Layout

```
packages/helios/
  src/                 TypeScript shim only (Backend impl, types)
  native/
    aether/            device.c, rm_object.c, ioctl.c, ...
    gaia/              vidmem.c, va_space.c, mapping.c, ...
    hermes/            channel.c, pushbuffer.c, qmd.c, method.c, ...
    chronos/           semaphore.c, fence.c, timeline.c
    hephaestus/        encode.c, control.c, regalloc.c, isa_*.c
    prometheus/        ir.c, codegen.c, kernels/*.c
    helios/            napi.c, backend.c, graph.c
    test/              from-scratch assert harness + per-layer tests
    build.mjs          globs the tree, one gcc invocation
```

## 4. Coding standards

These are binding, not aspirational.

0. **C11, freestanding of dependencies.** No libraries, no build system beyond `build.mjs` + `gcc`;
   nothing to install. Every layer is a directory; every file has a matching `.h` exposing only what
   the layer above needs. Anything not in the header is `static`. The only foreign headers we read are
   the vendor's `open-gpu-kernel-modules` SDK headers, for struct layouts and constants.

1. **One concept per file, ~300-line soft cap.** If a file needs internal section headers, it is two
   files. `helios_vk.c` reached **5,410 lines**; that is the anti-pattern being corrected, and it is
   what happens without this rule.

2. **Every file opens with a header block** stating what it is, why it exists, **what it deliberately
   does not do**, and the hardware fact it encodes.

3. **Comments explain *why*, and cite provenance.** Every magic constant carries the header and symbol
   it came from — `// NVC7C0_QMDV03_00_* — open-gpu-kernel-modules, clc7c0qmd.h`. An uncited number is
   a bug, because nobody can check it later.

4. **Strict downward dependencies**, enforced by a test that fails on any upward `#include`. `aether/`
   includes nothing of ours; `gaia/` may include `aether/`; and so on up.

5. **Known-answer tests at every layer.** Expected values come from algebra or the hardware spec —
   never from a second implementation. This is earned, not stylistic: X58's gradient-norm bug (silently
   ~half its true value, affecting gradient clipping) and X60's softmax bug both survived a full parity
   suite, because parity asks *"do these agree?"* — a question that goes quiet when both sides are
   wrong.

6. **A test must prove it executed the path it names.** X60's first suite "passed" every small shape
   while silently exercising the CPU fallback, because `DEFAULT_MIN_GPU_SIZE` routed around the GPU
   entirely. Assert which path ran.

7. **Hephaestus is validated by `nvdisasm` round-trip** — assemble, disassemble, compare text, for
   every instruction form we emit. `nvdisasm` is a dev-time oracle only, outside the training loop,
   consistent with `GOAL.md` ("external tools permitted only OUTSIDE the training loop").

8. **Tests are C too**, on a ~100-line assert harness written once in `native/test/`. Each layer's test
   binary links only that layer and the ones below it, which makes rule 4 self-enforcing: if `gaia`'s
   tests need something from `hermes`, the layering is wrong.

## 5. Phases

Each gate must pass on real hardware before the next phase begins.

| Phase | Work | Gate |
|---|---|---|
| **P0** | Spike: open device, allocate, build a channel, hand-encode a few SASS instructions, submit, read back | **A value produced by our own machine code appears in host memory.** Make-or-break. |
| **P1** | Aether + Gaia + Chronos | Allocate, map, host-write, read back, with a GPU-signalled fence |
| **P2** | Hermes | Hand-assembled vector-add over real buffers, correct at non-round sizes |
| **P3** | Hephaestus | `nvdisasm` round-trip identity across the full emitted instruction subset |
| **P4** | Prometheus + port 154 kernel generators (elementwise → reduction → nn → matmul → attention → optimizer) | `known-answer` suite green on hardware, per family |
| **P5** | Whole-stack parity, then delete Vulkan | `scripts/run_nvidia_gates.sh` 50/50 **and** `known-answer` green on a 3090 |

**The Vulkan code stays in the tree until P5** — not wired up, no fallback, no config flag. It is the
only independent implementation we have of 154 kernels, and it is worth keeping as a diffing oracle
while porting. It is deleted at P5, which is the hard-replace end state.

## 6. Replacement surface

| | |
|---|---|
| `packages/helios` today | ~40,200 LOC |
| Vulkan entry points to replace | 78 |
| `Backend` methods to satisfy | 71 |
| Kernel generators to port | 154 — 46 elementwise, 45 nn, 16 matmul, 9 matmul-coop, 9 reduction, 8 attention, 7 optimizer, 14 other |
| Kernel variants dispatched | 166 |

## 7. Risks

- **The control bits are the real danger.** Stall counts and dependency barriers are the least
  documented part of SASS, and getting them wrong yields silent corruption or a hung channel — not a
  clean error. That is precisely the failure signature of X58 and X60, which is why standard 5 is
  binding.
- **This is the largest single thing in the repo.** The assembler alone will likely exceed the current
  C addon.
- **Hard replace means no training until P5.** Recorded so it is not a surprise later.
- **No completeness guarantee on the ISA.** We can always encode more instructions; we can never prove
  we have them all — only that everything Alpha emits round-trips.

## 8. Inherited findings that still matter

- **X58** — the gradient norm was ~0.707× its true value for tensors ≥ 65,536 elements, weakening
  gradient clipping. The 3090 profile of record is `HELIOS_WG_SIZE=64` ([`RUNPOD.md`](RUNPOD.md)),
  below `STRIDE_WGS=256`, so **this was active on real 3090 training runs**. Historical `grad_norm`
  numbers should be read accordingly.
- **X60** — GPU softmax rows sum to ~4 instead of 1 on the software lane, at every size. Near-moot for
  this program since those kernels are being replaced, but the *method* that found it is standard 5.
- **X54/X55** — loss scaling was never connected to the cooperative backward path, destroying ~44% of
  backward gradient signal by step 125. Fixed but unverified on hardware; the new stack must not
  reintroduce an unscaled FP16 cast site.
