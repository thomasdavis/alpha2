/*
 * coverage.h — what this assembler can emit, what it cannot, and what the gap
 * costs.
 *
 * WHY IT EXISTS. The stack is written from scratch against an undocumented ISA,
 * so "we have not implemented that yet" is a permanent condition rather than a
 * temporary one, and the expensive version of it is finding out by accident.
 * Twice now a whole line of work has been shaped by a missing instruction that
 * nobody had written down:
 *
 *   - SHFL was missing, so every reduction went through shared memory with a
 *     block-wide barrier per step, and a 640-wide layer norm cost 65 us against
 *     a 26 us roofline. Nothing said "there is no warp shuffle"; the reduction
 *     just looked like a reasonable reduction.
 *   - LDG.E.128 IS encoded and has no caller, and the batched transposed-B GEMM
 *     is 2-3x slow for want of a 128-byte staging request. The capability and
 *     the problem it solves were in the same repository, unconnected.
 *
 * So this is a REGISTER, not a wish list: every entry names the instruction, its
 * state, and — the part that makes it worth maintaining — what is currently
 * PAYING for its absence, in microseconds where that is known.
 *
 * THE THREE STATES ARE DISTINCT AND MUST NOT BE CONFLATED:
 *
 *   HP_ISA_ENCODED    an encoder exists and is tested against a ptxas capture.
 *   HP_ISA_CAPTURED   the bits are known and written down, and no encoder emits
 *                     them. Cheap to finish — the expensive half is done.
 *   HP_ISA_MISSING    not captured. Needs a .cu, nvcc, and cuobjdump before any
 *                     of it is real.
 *
 * HOW TO MOVE SOMETHING FROM MISSING TO ENCODED, which is the same five steps
 * every instruction in this encoder went through:
 *   1. write the form in CUDA, at least TWICE with different registers and
 *      immediates — one capture cannot tell an operand's field from a constant
 *      that happens to sit in it (tools/shfl_capture.cu is the worked example);
 *   2. /usr/local/cuda-12.8/bin/nvcc -arch=sm_86 -cubin -o x.cubin x.cu
 *      && cuobjdump -sass x.cubin        (nvcc is NOT on PATH on the pod);
 *   3. derive each field from what MOVES between two captures;
 *   4. encode it, and assert the exact words in test/hephaestus_isa_test.c,
 *      including a check that no field reaches its neighbour;
 *   5. only then use it, and measure.
 *
 * ⚠️ A WRONG ENCODING DOES NOT FAULT. It executes a different instruction, or
 * reads registers nobody wrote. That is why step 4 is not optional and why
 * "captured" and "encoded" are separate states rather than a formality.
 *
 * THE EVIDENCE BASE is isa/sm86-catalogue.json — 728 instructions across 31
 * mnemonics that ptxas emitted for kernels of this shape. Anything in there
 * with no entry below is a gap nobody has looked at, and
 * packages/tests/audit-isa-coverage.mjs fails when one appears.
 */
#ifndef HELIOS_HEPHAESTUS_COVERAGE_H
#define HELIOS_HEPHAESTUS_COVERAGE_H

typedef enum {
  HP_ISA_ENCODED = 0,
  HP_ISA_CAPTURED = 1,
  HP_ISA_MISSING = 2,
} hp_isa_state;

typedef struct {
  const char *mnemonic; /* as nvdisasm prints it, so the catalogue can be diffed */
  hp_isa_state state;
  const char *blocks;   /* what the absence costs, or what the presence enables */
} hp_isa_entry;

/* The table, and its length. Ordered by what it would be worth to close. */
const hp_isa_entry *hp_isa_coverage(unsigned *count);

/* Convenience for a caller that wants to branch on availability rather than
 * read the table: 1 when the mnemonic is HP_ISA_ENCODED, 0 otherwise, and 0 for
 * a name that is not in the table at all. */
int hp_isa_have(const char *mnemonic);

/*
 * What a stub does when called.
 *
 * It ABORTS, and it prints the mnemonic, what the gap blocks, and the five
 * steps above. It does not return a plausible word: an encoder that returns
 * something wrong is indistinguishable from one that returns something right
 * until a matrix comes back finite, plausible and incorrect, which has happened
 * five times in this kernel already.
 *
 * Declared here rather than in sm86.h because nothing outside the assembler
 * should be calling it.
 */
void hp_isa_unimplemented(const char *mnemonic);

#endif /* HELIOS_HEPHAESTUS_COVERAGE_H */
