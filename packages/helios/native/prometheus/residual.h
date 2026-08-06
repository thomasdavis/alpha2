/*
 * residual.h — see residual.c.
 */
#ifndef PROMETHEUS_RESIDUAL_H
#define PROMETHEUS_RESIDUAL_H

#include "kernel.h"

/*
 * out = w * (x + residual) / sqrt(mean((x+residual)^2) + eps)
 *   slot 1 x, slot 2 residual, slot 3 w; scalars 1/N and eps.
 * One block of `elements` threads -- the reduction is block-local.
 */
unsigned pr_emit_residual_rms(hp_word *p, unsigned elements);

/*
 * out = residual + x * mask * scale
 *   slot 1 x, slot 2 residual, slot 3 mask; scalar the 1/(1-p) scale.
 */
unsigned pr_emit_residual_dropout(hp_word *p);

#endif /* PROMETHEUS_RESIDUAL_H */
