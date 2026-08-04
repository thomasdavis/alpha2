/*
 * optimizer.h — see optimizer.c.
 */
#ifndef PROMETHEUS_OPTIMIZER_H
#define PROMETHEUS_OPTIMIZER_H

#include "kernel.h"

/*
 * One AdamW step, in place over four tensors:
 *   slot 0  parameter (read and written)
 *   slot 1  gradient  (read)
 *   slot 2  m         (read and written)
 *   slot 3  v         (read and written)
 * and five scalars: 1-b1, 1-b2, the bias-corrected learning rate, epsilon, and
 * the weight decay. The bias corrections are folded into the learning rate by
 * the host, because they are the same for every element.
 */
unsigned pr_emit_adamw(hp_word *p);

#endif /* PROMETHEUS_OPTIMIZER_H */
