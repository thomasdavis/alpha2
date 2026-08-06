/* imma.h — see imma.c. A single 16x8x32 int8 tensor-core tile, to prove the
 * IMMA encoding + fragment layout compute correctly against a CPU reference. */
#ifndef PROMETHEUS_IMMA_H
#define PROMETHEUS_IMMA_H
#include "kernel.h"
unsigned pr_emit_imma_tile(hp_word *p);
#endif
