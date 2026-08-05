/*
 * colsum.h — out[c] = sum over rows of in[r][c].
 *
 * WHAT IT IS FOR: every bias-shaped gradient in the model. A layer norm's dw
 * and db, and every bias in a projection, are the same reduction — collapse the
 * token axis and keep the feature axis.
 *
 * WHY IT IS ITS OWN KERNEL, when two other routes already existed and both were
 * measured. `sum(x, 0)` takes the transpose route, which allocates a full
 * transposed copy of its input; a release only MARKS, so a step's peak is the
 * bytes ALLOCATED and two full-size transposes a layer is what exhausted the
 * card. The route that replaced it, `ones[1,R] @ x[R,C]`, allocates nothing
 * larger than the result and is honest arithmetic — and it is a GEMM with ONE
 * output row, so it fills a 32x128 tensor-core tile to one thirty-second and
 * falls off the HMMA path entirely.
 *
 * The measurement that made this a kernel: 74 calls of [1,1536] x [1536,640] a
 * step, 298 us each, 22.0 ms — **21% of the whole GPU step**, and the largest
 * single line in the profile by shape. It moves 3.9 MB and writes 2.5 KB, which
 * at this card's 448 GB/s is about 10 us of work. The GEMM was thirty times
 * that, and the profile could not see it because "matmul" averages this
 * together with the m1536 projections that run at 20 TFLOP/s.
 *
 * HOW: one block per 32 columns, 32 row-lanes deep. A warp covers 32 adjacent
 * columns so every global read is one coalesced 128-byte transaction; each lane
 * walks the row axis with a stride of 32 rows, accumulating in a register; the
 * 32 partials per column then meet in shared memory.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not split the row axis across
 * blocks. At the model's shape that would be the difference between 20 blocks
 * and 160 on a 46-SM card, and it is the obvious next move — but it needs
 * either a second pass or RED.E.ADD.F32 into a zeroed output, and a kernel that
 * is correct at 20 blocks is the one to measure that against.
 */
#ifndef HELIOS_PROMETHEUS_COLSUM_H
#define HELIOS_PROMETHEUS_COLSUM_H

#include "kernel.h"

/* Columns one block covers. A warp's width, so the lane index within a row is
 * `tid & 31` and the row-lane is `tid >> 5` — both shifts, no division. */
#define PR_COLSUM_COLS 32u

/* Row-lanes per column, i.e. how many partial sums meet in shared memory.
 * Thirty-two makes the block 1,024 threads, this hardware's maximum. */
#define PR_COLSUM_LANES 32u

#define PR_COLSUM_BLOCK (PR_COLSUM_COLS * PR_COLSUM_LANES)

/* Blocks needed to cover `cols` columns. */
unsigned pr_colsum_grid(unsigned cols);

/* Bytes of shared memory the kernel needs. */
unsigned pr_colsum_shared(void);

/*
 * The program. `rows` and `cols` are compiled in: the trip count and the row
 * stride are both immediates, and the tail predicate needs the column bound.
 *
 * Parameter 0 is the output [cols], parameter 1 the input [rows, cols].
 *
 * `product` adds a SECOND input at parameter 2 and reduces the elementwise
 * product instead: out[c] = sum_r a[r][c] * b[r][c].
 *
 * That exists because a layer norm's dw is sum_rows(g * xhat), and forming it
 * as a separate multiply costs a full read-add-write pass over a [1536,640]
 * tensor — 11.8 MB — to produce a value that is consumed immediately and only
 * as a sum. The multiply is one instruction inside a loop that is already
 * loading one of its two operands. It also removes the in-place trick the
 * caller used to avoid a third allocation, and with it the aliasing question
 * that made that trick need a paragraph of justification.
 */
unsigned pr_emit_column_sum(hp_word *prog, unsigned rows, unsigned cols,
                            int product);

#endif /* HELIOS_PROMETHEUS_COLSUM_H */
