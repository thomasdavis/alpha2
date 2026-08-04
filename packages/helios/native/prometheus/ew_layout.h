/*
 * ew_layout.h — the register and barrier layout shared by the two halves of the
 * element-wise generator.
 *
 * WHY IT EXISTS: elementwise.c decides the SHAPE of a per-element kernel -- work
 * out this thread's index, load what it needs, store the result -- and
 * elementwise_ops.c decides what each operation computes. Both have to agree on
 * which register holds the loaded value and which barrier tracks the load, and
 * that agreement is the whole interface between them.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it is private to this pair. Nothing outside
 * prometheus/ should know a register number.
 */
#ifndef PROMETHEUS_EW_LAYOUT_H
#define PROMETHEUS_EW_LAYOUT_H

#include "elementwise.h"

enum {
  R_INDEX = 0,
  R_IN_ADDR = 2,  /* R2:R3 */
  R_TID = 3,      /* consumed by the index IMAD, then reused as address high */
  R_VALUE = 4,
  R_ESIZE = 5,
  R_OUT_ADDR = 6, /* R6:R7 */
  R_RESULT = 8,
  R_B_ADDR = 10,  /* R10:R11 — the second input, for binary operations */
  R_B_VALUE = 12,
  R_SCALAR = 13,
  R_TEMP = 14,
  R_SCALAR2 = 15,
  R_TEMP2 = 16,
  R_SCALAR3 = 17,
  R_SCALAR4 = 18,
  R_TEMP3 = 19,
};

/* Bytes per element. Everything here is 32-bit. */
#define ELEMENT_BYTES 4

/* Scoreboard barriers. Both S2Rs share barrier 0 on purpose: the scoreboard
 * counts outstanding writes, so one barrier tracking two producers is exactly
 * what it is for, and ptxas does the same. Splitting them across two barriers
 * with a combined wait was tried and faults. */

#define BAR_INDEX 0
#define BAR_LOAD 1
#define BAR_MUFU 2
#define BAR_LOAD_B 3

/* Slot i of the constant bank always lands in SCALAR_REG[i]. */
extern const unsigned SCALAR_REG[HERMES_CBUF0_SCALAR_COUNT];

/* How many constants the operation reads, and whether its result arrives
 * through a barrier that the store has to wait on. */
unsigned pr_ew_scalars_read(pr_ew_op op);
int pr_ew_sets_barrier(pr_ew_op op);

/* The operation itself: R_RESULT = f(R_VALUE, R_INDEX). */
unsigned pr_ew_emit_op(hp_word *p, pr_ew_op op);

#endif /* PROMETHEUS_EW_LAYOUT_H */
