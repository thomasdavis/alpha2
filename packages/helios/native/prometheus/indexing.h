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

/* out[i][d] = table[ids[i]][d], table in the first input, ids in the second.
 * Launch one block per token, dim threads each. */
unsigned pr_emit_embedding(hp_word *p, unsigned dim);

#endif /* PROMETHEUS_INDEXING_H */
