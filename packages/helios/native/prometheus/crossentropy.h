/*
 * crossentropy.h — see crossentropy.c.
 */
#ifndef PROMETHEUS_CROSSENTROPY_H
#define PROMETHEUS_CROSSENTROPY_H

#include "kernel.h"

/*
 * out[r] = -log(softmax(logits[r])[targets[r]]), one block of `classes`
 * threads per row. Slot 1 logits, slot 2 targets (raw integers). Scalars:
 * log2(e) and ln(2).
 */
unsigned pr_emit_cross_entropy(hp_word *p, unsigned classes);

#endif /* PROMETHEUS_CROSSENTROPY_H */
