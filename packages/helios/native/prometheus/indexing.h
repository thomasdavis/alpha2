/*
 * indexing.h — see indexing.c.
 */
#ifndef PROMETHEUS_INDEXING_H
#define PROMETHEUS_INDEXING_H

#include "kernel.h"

/* out[c][r] = in[r][c]. Launch rows blocks of cols threads. */
unsigned pr_emit_transpose(hp_word *p, unsigned rows, unsigned cols);

/* Threads a transpose block runs — min(cols, 1024). Past that the launch is
 * invalid rather than slow; threads walk their columns in chunks instead. */
unsigned pr_transpose_block(unsigned cols);

/* Threads a row-copy block runs — min(W, 1024). slice, cat and broadcast all put
 * one thread on each column and walk the row in chunks past that. */
unsigned pr_row_block(unsigned W);

/* out[i] = in[offset + i*stride], both from the constant bank as integers. */
unsigned pr_emit_slice(hp_word *p);

/* out[b][h][t][d] = in[b][t][h][d]. T, H and D must be powers of two — the
 * index decomposition is shifts and masks, since sm_86 has no integer divide. */
/* out[r][c] = in[r][start + c]; `start` arrives in scalar 0. */
unsigned pr_emit_slice_rows(hp_word *p, unsigned W, unsigned srcW);

/* mode 0 tiles a vector down the rows, mode 1 spreads one value across each. */
unsigned pr_emit_broadcast(hp_word *p, unsigned mode, unsigned W);

/* out[r][start + c] = in[r][c]; `start` arrives in scalar 0. */
unsigned pr_emit_cat_rows(hp_word *p, unsigned W, unsigned dstW);

unsigned pr_emit_permute(hp_word *p, unsigned T, unsigned H, unsigned D);

/*
 * How many t-values one block covers.
 *
 * The kernel ran one thread per FEATURE and nothing else, so a block was D
 * threads — 64 for a 640-wide model over ten heads. An SM will hold sixteen
 * blocks, so that is 1,024 of its 1,536 thread slots and 5,120 blocks to
 * schedule for one permute; measured, the copy moved 2.6 MB in 24 us, which is
 * 109 GB/s against a 448 GB/s card.
 *
 * Giving a block several t-values costs one shift and one subtract in the
 * kernel and nothing anywhere else. Bounded by the 1,024-thread limit and by
 * what divides T, because a partial block would read and write past the plane.
 */
unsigned pr_permute_rows(unsigned T, unsigned D);

/* out[i][d] = table[ids[i]][d], table in the first input, ids in the second.
 * Launch one block per token, dim threads each. */
unsigned pr_emit_embedding(hp_word *p, unsigned dim);

#endif /* PROMETHEUS_INDEXING_H */
