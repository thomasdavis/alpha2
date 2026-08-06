# HMMA on sm_86 — the captured encoding

Every instruction in this encoder was built the same way: write the operation
in CUDA, compile it for sm_86, and read the bits out of `cuobjdump -sass`. The
IMAD note in `sm86.c` records one such capture. This is the same thing for the
tensor-core instruction, done ahead of implementing it, because whether HMMA is
reachable at all decides how fast this stack can ever go.

**It is reachable.** `nvcc 12.8` is installed on the 3070 box
(`/usr/local/cuda/bin/nvcc`), so the capture loop that produced every other
instruction is available for this one.

## Why it matters

Native runs its GEMM in scalar FP32 and reaches ~7.5% of the card's FP32 peak.
The Vulkan backend, on the SAME card and the SAME 105M model, reaches ~11,000
tok/s against native's ~1,188 — and the bulk of that 9x is not better
scheduling, it is that Vulkan's cooperative-matrix path runs on the TENSOR
CORES in f16 while native does not. The 3070's BF16/FP16 tensor peak is 40.6
TFLOP/s against 20.3 FP32, and dense tensor-core GEMMs reach a far higher
fraction of their peak than a scalar kernel does of its own.

So HMMA is the single largest remaining lever on the native backend, and
Vulkan is the existence proof that the hardware delivers it.

## The capture

```cuda
#include <cuda_fp16.h>
extern "C" __global__ void k(const half *A, const half *B, float *C) {
  float d[4] = {0,0,0,0};
  unsigned const *a = reinterpret_cast<unsigned const *>(A);
  unsigned const *b = reinterpret_cast<unsigned const *>(B);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  C[threadIdx.x] = d[0]+d[1]+d[2]+d[3];
}
```

    nvcc -arch=sm_86 -cubin -o m.cubin m.cu
    cuobjdump -sass m.cubin

yields

    HMMA.16816.F32 R8, R8, R6, RZ
        /* 0x000000060808723c */
        /* 0x004fe400000018ff */

Reading it against the field positions `word.h` already defines, and against
the IMAD capture in `sm86.c` (`opcode 0x424, dst at 16, srcA at 24, immediate
at 32, srcC at 64`):

| field | bits | value | meaning |
|---|---|---|---|
| opcode | low 12 | `0x23c` | HMMA.16816.F32 |
| dst | low @16 | `0x08` | R8 — the accumulator, four consecutive registers R8..R11 |
| srcA | low @24 | `0x08` | R8 — A fragment, four registers |
| srcB | low @32 | `0x06` | R6 — B fragment, two registers |
| srcC | high @0 | `0xff` | RZ, i.e. accumulate into the destination |

Note the destination and srcA are the same register here only because the
compiler chose to accumulate in place; they are independent fields.

## The fragment layout — MEASURED on the 3070, not read from the docs

Settled ahead of implementing the kernel, because getting it wrong is silent.
A CUDA probe filled A (16x16) and B (8x16) with distinct exactly-representable
values, loaded the fragments according to the layout below, ran the real
`mma.sync`, and compared all 128 outputs against a host reference:

    m16n8k16 row.col fragment layout: worst abs error 0 across 128 elements
    LAYOUT CONFIRMED on this GPU

Exact, not approximate — the inputs are small integers exact in f16 and the
accumulation is f32.

With `g = lane >> 2` (the lane's group, 0..7) and `l = lane & 3` (its position
in that group):

| fragment | register | holds |
|---|---|---|
| A (4 regs, f16x2) | a0 | rows `g`,   k = `2l`,   `2l+1` |
|                   | a1 | rows `g+8`, k = `2l`,   `2l+1` |
|                   | a2 | rows `g`,   k = `2l+8`, `2l+9` |
|                   | a3 | rows `g+8`, k = `2l+8`, `2l+9` |
| B (2 regs, f16x2) | b0 | col `g`, k = `2l`,   `2l+1` |
|                   | b1 | col `g`, k = `2l+8`, `2l+9` |
| C/D (4 regs, f32) | d0,d1 | row `g`,   cols `2l`, `2l+1` |
|                   | d2,d3 | row `g+8`, cols `2l`, `2l+1` |

So A is read row-major and B column-major — which is why the operation is
`row.col`, and why it wants B TRANSPOSED. That is worth noting against
matmul.c: the fused `matmulTransposed` was refuted for the scalar kernel
because transposed B made the loads uncoalesced, but a tensor-core GEMM wants
exactly that orientation, and the tile is staged through shared memory where
the global access pattern is decoupled from the fragment's. The refutation
does not carry over.

The probe source is reproduced in the git history of this file's commit; it
needs only nvcc, which is installed on the box.

## What is NOT in this capture, and is the actual work

The encoding is the easy half. `m16n8k16` is a WARP-level instruction: the
32 lanes of a warp cooperatively hold one 16x8 output tile, and each lane owns
a specific, non-obvious subset of A, B and C. Getting the bits right and the
FRAGMENT LAYOUT wrong produces a finite, plausible, wrong matrix — the same
failure mode the first tiled attempt had, and it will pass any test that only
checks for NaNs.

So the implementation order should be:

1. `hp_hmma(dst, a, b, c, ctrl)` in `sm86_float.c`, encoding the above.
2. ~~Establish the lane-to-element mapping by measurement.~~ DONE — see the
   table above, confirmed exactly on this GPU. A bare-metal probe under
   `tools/` is still worth writing, but it is now a VERIFICATION of a known
   layout rather than a discovery exercise, which is a much smaller job.
3. Only then wire it into `matmul.c`, which already has the shared-memory
   staging and register-tile structure the fragments need — that was the
   prerequisite, and it now ships at four rows.

Two further requirements the rest of the stack does not yet meet:

- **f16 storage.** The tensor cores consume half-precision inputs. There is a
  `cast` kernel exported already; the GEMM would need its operands in f16 with
  the accumulator staying f32 (which `.f32.f16.f16.f32` gives).
- **Register budget.** A 16x8x16 fragment set is 4 + 2 + 4 registers per lane
  before addressing. `helios_program.regs` now carries a per-program
  declaration, so this can ask for what it needs without taxing every other
  kernel — that hook exists for exactly this.
