/*
 * dropout.h — see dropout.c.
 */
#ifndef PROMETHEUS_DROPOUT_H
#define PROMETHEUS_DROPOUT_H

#include "kernel.h"

/*
 * mask[i] = hash(seed, counter, i) < threshold ? 0 : scale.
 * Scalars: seed, counter, threshold (all read as raw 32-bit integers), scale.
 * The threshold is p * 2^32, computed on the host.
 */
unsigned pr_emit_dropout_mask(hp_word *p);

#endif /* PROMETHEUS_DROPOUT_H */
