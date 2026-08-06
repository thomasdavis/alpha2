/*
 * sm86_stub.c — see coverage.h.
 *
 * The register of what this assembler can and cannot emit, and loud stubs for
 * the forms that are named but not encoded.
 *
 * WHY A STUB RATHER THAN NOTHING. A missing encoder is a link error at best and
 * an absent thought at worst — the reduction went through shared memory for
 * months because nobody had written down that there was no SHFL. A stub that
 * aborts with the capture recipe turns "why is this slow" into "this needs
 * twenty minutes with nvcc", which is a different kind of problem.
 */
#include "coverage.h"
#include "sm86.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * ORDERED BY WHAT IT WOULD BE WORTH TO CLOSE, most first, because the order is
 * the only editorial content a table like this can carry.
 *
 * The `blocks` strings carry MEASUREMENTS where there are any. A gap with a
 * microsecond figure beside it is a decision; a gap with an adjective beside it
 * is a preference.
 */
static const hp_isa_entry TABLE[] = {
    /* ---- encoded, and load-bearing ------------------------------------- */
    {"HMMA", HP_ISA_ENCODED,
     "the tensor cores. 45.5 TFLOP/s from registers against 20.3 FP32; the "
     "GEMM sustains 15-21 at the model's shapes"},
    {"LDSM", HP_ISA_ENCODED,
     "ldmatrix: a whole mma fragment in one shared load, sixteen down to six "
     "per k-step. +0.9% end to end, kept"},
    {"SHFL", HP_ISA_ENCODED,
     "warp reductions without a barrier. Closed 2026-08-06: layerNorm 65.1 -> "
     "56.4 us, softmax [15360,64] 43.1 -> 31.1. ONLY the both-immediate form "
     "(0xf89) is encoded — see SHFL.reg below"},
    {"RED", HP_ISA_ENCODED,
     "RED.E.ADD.F32 only. The embedding gradient's scatter, dW[ids[i]] += g[i], "
     "which replaced a 24 GFLOP matmul against a one-hot"},
    {"LDG", HP_ISA_ENCODED, "global load, 32/64/128-bit — but see LDG.E.128"},
    {"STG", HP_ISA_ENCODED, "global store"},
    {"LDS", HP_ISA_ENCODED, "shared load, .X4-scaled addressing"},
    {"STS", HP_ISA_ENCODED, "shared store"},
    {"F2FP", HP_ISA_ENCODED, "f32 pair -> packed f16, the GEMM's operand path"},
    {"HADD2", HP_ISA_ENCODED, "packed f16 add; also the f16 unpack"},
    {"FADD", HP_ISA_ENCODED, "f32 add"},
    {"FMUL", HP_ISA_ENCODED, "f32 multiply"},
    {"FFMA", HP_ISA_ENCODED, "f32 fused multiply-add"},
    {"FMNMX", HP_ISA_ENCODED, "f32 min/max — the softmax's row maximum"},
    {"MUFU", HP_ISA_ENCODED, "EX2 LG2 RCP RSQ SQRT TANH"},
    {"IADD3", HP_ISA_ENCODED, "integer add, register and immediate forms"},
    {"IMAD", HP_ISA_ENCODED, "integer multiply-add, incl. WIDE for addresses"},
    {"ISETP", HP_ISA_ENCODED, "integer compare to a predicate; LT EQ LE GT NE GE"},
    {"LOP3", HP_ISA_ENCODED, "any function of three inputs from a truth table"},
    {"SHF", HP_ISA_ENCODED, "funnel shift, right only"},
    {"MOV", HP_ISA_ENCODED, "register, immediate and const-bank forms"},
    {"S2R", HP_ISA_ENCODED, "special registers: tid, ctaid"},
    {"BAR", HP_ISA_ENCODED, "BAR.SYNC only — see BAR.RED"},
    {"BRA", HP_ISA_ENCODED, "relative branch; the only control flow there is"},
    {"NOP", HP_ISA_ENCODED, "and it carries a control field, so it is how a "
     "standalone barrier wait is placed"},
    {"EXIT", HP_ISA_ENCODED, "kernel exit"},

    /* ---- captured or encoded, and UNUSED — the cheapest wins ------------ */
    {"LDG.E.128", HP_ISA_CAPTURED,
     "ENCODED WITH NO CALLER. A thread would own four consecutive k instead of "
     "two, which is the change the staging assignment needs. Directly relevant: "
     "batched transposed-B is 2-3x slow because a staged tile row is MMA_K=16 k "
     "= 64 bytes and a warp therefore makes four requests where the "
     "untransposed layout makes one. Worth 2.1 ms of a 79 ms step"},
    {"SHFL.reg", HP_ISA_CAPTURED,
     "the register-lane (0x589) and register-c (0x989) forms. Captured in "
     "tools/shfl_capture.cu and named in isa.h; not encoded because a reduction "
     "only ever shuffles by a constant. Needed for a segment width chosen at "
     "run time, which is what a warp reduction over a width not divisible by 32 "
     "would want — those widths currently keep the halving tree"},

    /* ---- missing, and each one gates something measured ----------------- */
    {"LDGSTS", HP_ISA_CAPTURED,
     "cp.async: global -> shared WITHOUT passing through registers. ENCODED "
     "2026-08-06 with LDGDEPBAR and DEPBAR, and NOT YET CALLED — it is the gate "
     "for f16-in-memory staging, because cp.async copies bytes and cannot "
     "convert f32 on the way. The GEMM's staging is load-to-register, pack, "
     "store-to-shared, and this deletes two thirds of that. Double buffering "
     "was measured at 3-5% and declined, but that measured BARRIERS, not the "
     "register round trip. ENCODED + WIRING DECODED 2026-08-06 (LDGDEPBAR sets "
     "wbar 0; see sm86_mem.c) — ready to thread into the GEMM staging, which the "
     "SASS says would delete 28 of the 42 k-step instructions. One offset "
     "field's granularity still unproven"},
    {"LDGDEPBAR", HP_ISA_ENCODED,
     "closes a cp.async group AND must SET WRITE BARRIER 0 (ptxas ctrl "
     "0x000e2200) to arm async scoreboard SB0 — pass hp_ctrl_setbar(0). Decoded "
     "2026-08-06; a safe control leaves SB0 unarmed and DEPBAR returns at once"},
    {"DEPBAR", HP_ISA_ENCODED,
     "waits until at most N cp.async groups are outstanding. N=0 drains and is "
     "no better than a synchronous load; the instruction earns its place at 1 "
     "or 2, which is what puts stage N+1 in flight while stage N is consumed"},
    {"HFMA2", HP_ISA_MISSING,
     "packed f16 fused multiply-add. Two lanes of arithmetic per instruction, "
     "and the precondition for f16 activations in MEMORY rather than only in "
     "the tensor fragments. The step moves ~16 GB and its non-GEMM half is at "
     "the bandwidth ceiling (340-417 GB/s of 448), so halving the bytes is the "
     "one lever left with a factor behind it. In the catalogue"},
    {"FSETP", HP_ISA_MISSING,
     "float compare to a predicate. Every float comparison in prometheus is "
     "currently an FMNMX and a subtract, or an integer compare on the bit "
     "pattern. In the catalogue"},
    {"FSEL", HP_ISA_MISSING,
     "branchless float select. Pairs with FSETP; today the same thing is a "
     "predicated pair of instructions. In the catalogue"},
    {"I2FP", HP_ISA_MISSING,
     "integer -> float. There is no conversion in the encoder at all, so a "
     "count or an index cannot become a scale factor inside a kernel — which is "
     "why every mean divides by a HOST-computed reciprocal passed in the "
     "constant bank. In the catalogue"},
    {"F2I", HP_ISA_MISSING,
     "float -> integer. Blocks argmax returning an index, and any gather whose "
     "index is computed from data. In the catalogue"},
    {"ATOMG", HP_ISA_MISSING,
     "atomics that RETURN the previous value. RED covers fire-and-forget add "
     "only, so a multi-block reduction still needs a second pass and a second "
     "launch. In the catalogue"},
    {"VOTE", HP_ISA_MISSING,
     "warp ballot. With POPC it makes a stream-compaction primitive; nothing "
     "here needs one yet. In the catalogue"},
    {"REDUX", HP_ISA_MISSING,
     "a whole-warp reduction in ONE instruction, integer only on this "
     "architecture — so it does not replace the SHFL butterfly for floats. "
     "Recorded so nobody reaches for it expecting that"},
    {"BAR.RED", HP_ISA_MISSING,
     "named and reducing barriers. BAR.SYNC synchronises the whole block, so a "
     "producer warp cannot wait on a consumer warp alone; that is what a "
     "warp-specialised GEMM pipeline needs"},
    {"PRMT", HP_ISA_MISSING, "byte permute. A cheaper f16 pack and unpack than "
     "the F2FP/HADD2 pair in some layouts; not measured"},
    {"S2UR", HP_ISA_MISSING,
     "special register into a UNIFORM register. Uniform datapath — one copy of "
     "a block-invariant value per warp instead of one per lane. The GEMM "
     "recomputes block-invariant addresses in every lane. In the catalogue"},
    {"ULDC", HP_ISA_MISSING,
     "uniform const-bank load. Same story as S2UR: kernel parameters are read "
     "per-lane today. In the catalogue"},
};

const hp_isa_entry *hp_isa_coverage(unsigned *count) {
  if (count) *count = (unsigned)(sizeof TABLE / sizeof TABLE[0]);
  return TABLE;
}

int hp_isa_have(const char *mnemonic) {
  const unsigned n = (unsigned)(sizeof TABLE / sizeof TABLE[0]);
  for (unsigned i = 0; i < n; i++)
    if (strcmp(TABLE[i].mnemonic, mnemonic) == 0)
      return TABLE[i].state == HP_ISA_ENCODED;
  return 0;
}

void hp_isa_unimplemented(const char *mnemonic) {
  const unsigned n = (unsigned)(sizeof TABLE / sizeof TABLE[0]);
  const char *blocks = "(not in the coverage table — add it)";
  for (unsigned i = 0; i < n; i++)
    if (strcmp(TABLE[i].mnemonic, mnemonic) == 0) blocks = TABLE[i].blocks;

  fprintf(stderr,
          "\nhephaestus: %s is NOT ENCODED.\n"
          "  what it is for: %s\n"
          "\n"
          "  This aborts rather than returning a word, because a wrong encoding\n"
          "  does not fault on this hardware — it executes a different\n"
          "  instruction and returns a finite, plausible, wrong answer.\n"
          "\n"
          "  To add it (the five steps every encoder here went through):\n"
          "    1. write the form in CUDA at least TWICE, with different\n"
          "       registers and immediates — tools/shfl_capture.cu is the\n"
          "       worked example, eleven kernels for one instruction;\n"
          "    2. /usr/local/cuda-12.8/bin/nvcc -arch=sm_86 -cubin -o x.cubin x.cu\n"
          "       && /usr/local/cuda-12.8/bin/cuobjdump -sass x.cubin\n"
          "       (nvcc is not on PATH on the pod);\n"
          "    3. derive every field from what MOVES between two captures;\n"
          "    4. assert the exact words in test/hephaestus_isa_test.c, and\n"
          "       assert that no field reaches its neighbour;\n"
          "    5. flip its row in sm86_stub.c to HP_ISA_ENCODED, then measure.\n\n",
          mnemonic, blocks);
  abort();
}

/*
 * THE STUBS.
 *
 * Only for forms a caller might plausibly reach for TODAY and be surprised to
 * find absent. An instruction nobody is about to call needs a table row, not a
 * function — a stub that exists only to abort is a maintenance cost with no
 * reader.
 *
 * Each returns hp_word so it can be dropped into an emitter unchanged once it
 * is real; none of them returns.
 */
hp_word hp_shfl_reg(unsigned mode, unsigned dst, unsigned src, unsigned laneReg,
                    unsigned cImm, hp_control c) {
  (void)mode; (void)dst; (void)src; (void)laneReg; (void)cImm; (void)c;
  hp_isa_unimplemented("SHFL.reg");
  return (hp_word){0, 0};
}

hp_word hp_hfma2(unsigned dst, unsigned srcA, unsigned srcB, unsigned srcC,
                 hp_control c) {
  (void)dst; (void)srcA; (void)srcB; (void)srcC; (void)c;
  hp_isa_unimplemented("HFMA2");
  return (hp_word){0, 0};
}

hp_word hp_fsetp(unsigned destPred, unsigned srcA, unsigned srcB, unsigned cmp,
                 hp_control c) {
  (void)destPred; (void)srcA; (void)srcB; (void)cmp; (void)c;
  hp_isa_unimplemented("FSETP");
  return (hp_word){0, 0};
}

hp_word hp_i2f(unsigned dst, unsigned src, hp_control c) {
  (void)dst; (void)src; (void)c;
  hp_isa_unimplemented("I2FP");
  return (hp_word){0, 0};
}

hp_word hp_f2i(unsigned dst, unsigned src, hp_control c) {
  (void)dst; (void)src; (void)c;
  hp_isa_unimplemented("F2I");
  return (hp_word){0, 0};
}
