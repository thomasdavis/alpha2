/*
 * program.h — generated kernels, kept.
 *
 * WHAT: a cache from (operation, shape) to an assembled program and the launch
 * shape that goes with it.
 *
 * WHY A CACHE AND NOT A COMPILE PER CALL: every emitter here specialises on the
 * shape -- K becomes an immediate in matmul's loop bound, the reduction tree is
 * unrolled to the element count -- which is what makes the generated code good
 * and also what would make regenerating it per call absurd. A training step
 * runs the same dozen shapes thousands of times. Emitting is cheap, but not
 * free, and doing it once per shape rather than once per call is the difference
 * between a fixed startup cost and a per-launch tax.
 *
 * WHY THE LAUNCH SHAPE IS CACHED WITH THE CODE: they are chosen together. A
 * matmul emitted for M x N x K only makes sense launched as M blocks of N
 * threads, and separating them would let a caller pair a program with a grid
 * that does not match it -- a class of bug with no symptom except wrong
 * numbers.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no eviction. The set of shapes a model uses
 * is fixed after the first step, so a cache that fills is a cache that was
 * sized wrong, and reporting that is more useful than quietly evicting an entry
 * that is about to be asked for again.
 */
#ifndef HELIOS_PROGRAM_H
#define HELIOS_PROGRAM_H

#include "../prometheus/kernel.h"

/*
 * What kind of kernel, independent of shape.
 *
 * The element-wise operations reuse prometheus's own enum rather than
 * restating it: a second list of the same operations is a second thing to keep
 * in step, and they would drift the first time one was added to only one.
 */
typedef enum {
  HL_ELEMENTWISE, /* arg0 = pr_ew_op */
  HL_REDUCE,      /* arg0 = pr_red_op, arg1 = elements */
  HL_NORMALIZE,   /* arg0 = pr_norm_op, arg1 = elements */
  HL_MATMUL,      /* arg0 = M, arg1 = N, arg2 = K */
  HL_TRANSPOSE,   /* arg0 = rows, arg1 = cols */
  HL_SLICE_ROWS,  /* arg0 = out width, arg1 = source width */
  HL_BROADCAST,   /* arg0 = mode (0 tile, 1 row), arg1 = width */
  HL_PERMUTE,     /* arg0 = T, arg1 = H, arg2 = D — swaps the middle two axes */
  HL_EMBEDDING,   /* arg0 = dim */
  HL_SLICE,
  HL_CAUSAL_MASK, /* arg0 = cols */
  HL_MASKED_FILL,
  HL_CAST_TO_F16,
  HL_CAST_TO_F32,
  HL_DROPOUT,
  HL_CROSS_ENTROPY,   /* arg0 = classes */
  HL_RESIDUAL_RMS,    /* arg0 = elements */
  HL_RESIDUAL_DROPOUT,
  HL_ADAMW,
  HL_REDUCE_PARTIAL, /* arg0 = pr_red_op, arg1 = elements per block */
  HL_KIND_COUNT
} helios_kind;

typedef struct {
  helios_kind kind;
  NvU32 arg0, arg1, arg2;
} helios_key;

typedef struct {
  helios_key key;
  hp_word code[PR_MAX_INSTRUCTIONS];
  unsigned count;
  NvU32 blockX, gridX, sharedBytes;
  /* A second grid dimension, for kernels that take a batch index from ctaid.y.
   * Zero means one -- most kernels are one-dimensional and should not have to
   * say so. */
  NvU32 gridY;
  int used;
} helios_program;

/*
 * Look the key up, emitting on a miss. Returns NULL only if the cache is full
 * or the key names a kind that does not exist -- both of which are programming
 * errors rather than conditions to handle, and both of which are silent if the
 * caller ignores the return.
 */
const helios_program *helios_program_get(helios_key key);

/* How many distinct programs have been generated. For tests, and for anyone
 * wondering whether a step is regenerating code it should be reusing. */
unsigned helios_program_count(void);

/* Drop everything. Only for tests -- a running backend has no reason to. */
void helios_program_reset(void);

#endif /* HELIOS_PROGRAM_H */
