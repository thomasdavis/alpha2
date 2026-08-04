/*
 * kernel.h — what a kernel is, from the harness's point of view.
 *
 * WHAT: a description of one kernel: how to assemble it, how to launch it, what
 * to feed it, and how to tell whether the GPU did the right thing.
 *
 * WHY a descriptor rather than a function per kernel: there are on the order of
 * a hundred and fifty kernels to port, and they differ in a handful of ways —
 * the operation in the middle, the launch geometry, the expected answer.
 * Everything else (channel, engine init, descriptor, submission, fence) is
 * identical, and writing it out once per kernel would be writing the same
 * hundred lines a hundred and fifty times. So the shared part is a runner and
 * the varying part is this struct.
 *
 * WHY the checker returns a STRING rather than a bool: "wrong" is not a useful
 * failure. A checker that can say "wrote 16 of 64 slots" turns a red test into
 * a diagnosis, and that difference has already been worth hours here — a fence
 * writing into the output array produced exactly that message and it named the
 * bug on sight.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no dependencies between kernels, no
 * multi-launch sequences, no timing. Correctness first; a kernel that is not
 * known-correct is not worth measuring.
 */
#ifndef HELIOS_PROMETHEUS_KERNEL_H
#define HELIOS_PROMETHEUS_KERNEL_H

#include "../hephaestus/sm86.h"
#include "../hermes/qmd.h"

/* Elements a kernel processes, and the input value at each index. Kept small
 * enough that a checker can walk the whole output. */
#define PR_N 64

/* Upper bound on the instructions any kernel emits, so a caller can size a
 * buffer without knowing which kernel it is about to build. Deliberately
 * generous; the cost of being wrong is a stack overwrite. */
#define PR_MAX_INSTRUCTIONS 64

typedef struct {
  const char *name;

  /* Assemble the kernel. `out` and `in` are GPU addresses, passed for kernels
   * that embed them as immediates; kernels that read the constant bank ignore
   * both and pick the pointers up at run time. Returns the instruction count. */
  unsigned (*build)(hp_word *prog, NvU64 out, NvU64 in);

  /* Launch geometry. Zero means the registry default, which is what almost
   * every elementwise kernel wants; a kernel with different needs says so. */
  NvU32 blockX, gridX;

  /* Prepare the inputs. Binary kernels use both; unary kernels ignore `b`;
   * kernels that read nothing leave this NULL. */
  void (*fill)(volatile NvU32 *a, volatile NvU32 *b);

  /* Inspect the output. NULL means correct; anything else is the reason, and
   * should say what was wrong rather than that something was. */
  const char *(*check)(const volatile NvU32 *o);

  /* A scalar operand, placed in the constant bank. Kernels that take one --
   * scale, and the base conversions inside exp and log -- read it from there,
   * which is how a real kernel receives a scalar. Last in the struct so the
   * registry table reads name, code, geometry, data, expectation. */
  float scalar, scalar2;

  /* Seed the OUTPUT before launch, for kernels that read what they write.
   * Accumulate-style operations are only meaningfully tested against a known
   * prior value; clearing to zero would make an accumulate indistinguishable
   * from an assign. */
  void (*seed)(volatile NvU32 *out);

  /* Shared memory the kernel needs, in bytes. Declared in the QMD; a kernel
   * that reads or writes shared memory it did not ask for faults. */
  NvU32 sharedBytes;
} pr_kernel;

/* The registry. Every kernel the stack can run, in the order they are tested. */
const pr_kernel *pr_kernels(unsigned *count);

/* Helpers shared by the checkers: float and its bit pattern are the same thing
 * to the GPU and different things to C, and memcpy is the only conversion that
 * is not undefined behaviour. */
NvU32 pr_f2u(float f);
float pr_u2f(NvU32 u);

#endif /* HELIOS_PROMETHEUS_KERNEL_H */
