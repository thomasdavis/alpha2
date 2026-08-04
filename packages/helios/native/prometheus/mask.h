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

#endif /* PROMETHEUS_MASK_H */
