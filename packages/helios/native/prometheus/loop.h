/*
 * loop.h — see loop.c.
 */
#ifndef PROMETHEUS_LOOP_H
#define PROMETHEUS_LOOP_H

#include "kernel.h"

/* out[i] = in[i], with a zero-distance branch in the middle that must do
 * nothing. Separates a malformed BRA from a wrong branch distance. */
typedef enum {
  PR_BRANCH_PLAIN,      /* unpredicated, zero distance */
  PR_BRANCH_PREDICATED, /* predicated, zero distance */
  PR_BRANCH_SKIP,       /* predicated, forward over one instruction */
} pr_branch_mode;

unsigned pr_emit_branch_nop(hp_word *p, pr_branch_mode mode);

/* out[i] = in[i] * trips, computed the long way round on purpose. */
unsigned pr_emit_loop_scale(hp_word *p, unsigned trips);

#endif /* PROMETHEUS_LOOP_H */
