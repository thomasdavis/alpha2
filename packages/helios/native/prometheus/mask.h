/*
 * mask.h — see mask.c.
 */
#ifndef PROMETHEUS_MASK_H
#define PROMETHEUS_MASK_H

#include "kernel.h"

/* out[r][c] = c > r ? -inf : in[r][c]. Launch rows blocks of cols threads. */
unsigned pr_emit_causal_mask(hp_word *p, unsigned cols);

/* out[i] = mask[i] ? scalar : in[i], mask in the second input buffer. */
unsigned pr_emit_masked_fill(hp_word *p);

/*
 * The same, with the mask TILED rather than materialised.
 *
 * The causal mask is [T,T] — 16 KB at T=64, and permanently resident in L2 —
 * while the scores it applies to are [B,H,T,T]. The untiled form needs them the
 * same size, so every call ran `expand` first: a 3.93 MB allocation and a full
 * write to reproduce 16 KB of data, thirty-six times a step, in a stack where a
 * step's PEAK is the bytes it allocates rather than the bytes it holds.
 *
 * `maskWrap` is the mask's element count and must be a power of two, so the
 * wrap is one AND rather than a division. Callers with any other mask size keep
 * the untiled form.
 */
unsigned pr_emit_masked_fill_tiled(hp_word *p, unsigned maskWrap);

#endif /* PROMETHEUS_MASK_H */
