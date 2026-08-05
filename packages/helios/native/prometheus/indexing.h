/*
 * indexing.h — see indexing.c.
 */
#ifndef PROMETHEUS_INDEXING_H
#define PROMETHEUS_INDEXING_H

#include "kernel.h"

/* out[c][r] = in[r][c]. Launch rows blocks of cols threads. */
unsigned pr_emit_transpose(hp_word *p, unsigned rows, unsigned cols);

/* out[i] = in[offset + i*stride], both from the constant bank as integers. */
unsigned pr_emit_slice(hp_word *p);

/* out[b][h][t][d] = in[b][t][h][d]. T, H and D must be powers of two — the
 * index decomposition is shifts and masks, since sm_86 has no integer divide. */
unsigned pr_emit_permute(hp_word *p, unsigned T, unsigned H, unsigned D);

/* out[i][d] = table[ids[i]][d], table in the first input, ids in the second.
 * Launch one block per token, dim threads each. */
unsigned pr_emit_embedding(hp_word *p, unsigned dim);

#endif /* PROMETHEUS_INDEXING_H */
