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

/*
 * The largest block a reduction tree can use.
 *
 * Bounded by the hardware's 1024 threads per block, and by shared memory: the
 * tree needs four bytes per thread, so 1024 threads is 4 KiB, well inside the
 * SM's window. A reduction over more than this needs two passes, which is the
 * dispatcher's job rather than the kernel's.
 */
#define PR_MAX_BLOCK 1024u

/*
 * Upper bound on the instructions any kernel emits, so a caller can size a
 * buffer without knowing which kernel it is about to build.
 *
 * 64 was not enough and the way it failed is the point: softmax reduces twice,
 * each tree is six unrolled steps of six instructions, and the builder wrote
 * straight past the end of the caller's array. The well-formedness test checked
 * `count <= PR_MAX_INSTRUCTIONS` -- AFTER the builder had already overrun -- so
 * the check could not catch the thing it existed to catch, and the failure
 * arrived as a stack smash instead of a message.
 *
 * The bound is now generous, and the test allocates twice it with a sentinel
 * behind, so an overrun is DETECTED rather than fatal.
 */
#define PR_MAX_INSTRUCTIONS 256

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
  /*
   * Judges the tensors a kernel writes BESIDES the output buffer.
   *
   * AdamW writes three tensors -- the parameter, and both moment estimates --
   * and a checker that reads only the output would pass a kernel that computed
   * the right parameter while storing the moments to the wrong place. The
   * parameter depends on both moments, so wrong VALUES were already caught;
   * wrong destinations were not, and on the second step they become wrong
   * values too. Given the two input buffers a kernel may have written back.
   */
  const char *(*checkAux)(const volatile NvU32 *b, const volatile NvU32 *c);

  /* A scalar operand, placed in the constant bank. Kernels that take one --
   * scale, and the base conversions inside exp and log -- read it from there,
   * which is how a real kernel receives a scalar. Last in the struct so the
   * registry table reads name, code, geometry, data, expectation. */
  float scalar, scalar2, scalar3, scalar4, scalar5, scalar6;
  /*
   * Scalar slots the kernel reads as raw 32-bit INTEGERS rather than floats.
   * A non-zero entry here overrides the float of the same index.
   *
   * The constant bank is untyped -- it is 4 KiB of words -- so this is not a
   * second mechanism, only a second way of spelling the same one. It exists
   * because passing a seed through a float would leave the compiler free to
   * canonicalise a bit pattern that happens to be a signalling NaN, and a
   * random-number seed is exactly the kind of value that eventually is one.
   * Zero means "not set", which is unambiguous: a raw zero and a float zero are
   * the same word.
   */
  NvU32 rawScalar[HERMES_CBUF0_SCALAR_COUNT];

  /* Seed the OUTPUT before launch, for kernels that read what they write.
   * Accumulate-style operations are only meaningfully tested against a known
   * prior value; clearing to zero would make an accumulate indistinguishable
   * from an assign. */
  /* Fills the THIRD input buffer, for the kernels that need four tensors --
   * the optimizer step and the fused residual paths. Separate from `fill`
   * rather than widening it, so the thirty-odd kernels that will never use a
   * third input do not have to mention it. */
  void (*fillC)(volatile NvU32 *c);
  void (*seed)(volatile NvU32 *out);

  /* Shared memory the kernel needs, in bytes. Declared in the QMD; a kernel
   * that reads or writes shared memory it did not ask for faults. */
  NvU32 sharedBytes;
  /* How many elements each thread handles. Zero means one, so the thirty-odd
   * kernels that handle exactly one do not have to say so. The cast kernels
   * handle a pair, which is why the launch-size check multiplies by this rather
   * than assuming threads and elements are the same count. */
  NvU32 elementsPerThread;
  /*
   * How many elements the launch is expected to span. Zero means PR_N.
   *
   * Not the same as the output size, and the difference is worth naming. Cross
   * entropy launches one thread per LOGIT and writes one loss per row, so its
   * work domain is the logits; slice launches one thread per output element and
   * reads from a larger source, so its work domain is the output. What the
   * check needs is the domain the launch is meant to cover, which only the
   * kernel knows.
   */
  NvU32 workElements;
  /*
   * How many leading output slots the checker actually inspects. Zero means
   * workElements.
   *
   * This is not documentation. After a kernel passes on hardware, the runner
   * perturbs each of these slots in turn and requires the checker to REJECT the
   * result -- so a wrong number here fails immediately, either because a slot
   * it claimed was checked is not, or because the count is short and a real
   * checker catches more than was declared. A checker that passed everything
   * would be caught the first time it ran.
   */
  NvU32 checkedElements;
} pr_kernel;

/* The registry. Every kernel the stack can run, in the order they are tested. */
const pr_kernel *pr_kernels(unsigned *count);

/* Helpers shared by the checkers: float and its bit pattern are the same thing
 * to the GPU and different things to C, and memcpy is the only conversion that
 * is not undefined behaviour. */
NvU32 pr_f2u(float f);
float pr_u2f(NvU32 u);

#endif /* HELIOS_PROMETHEUS_KERNEL_H */
