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

/* The gradient, same block shape: (softmax(z) - onehot(target)) * scale, in
 * one pass over the logits and with no one-hot tensor anywhere. */
unsigned pr_emit_cross_entropy_backward(hp_word *p, unsigned classes);

/* Threads a cross-entropy block runs — min(classes, 1024). Past that the launch
 * is invalid, and the shared-memory request would be the whole per-block budget.
 * Threads cover the vocabulary in chunks instead. */
unsigned pr_cross_entropy_block(unsigned classes);

#endif /* PROMETHEUS_CROSSENTROPY_H */
