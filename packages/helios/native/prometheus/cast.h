/*
 * cast.h — see cast.c.
 */
#ifndef PROMETHEUS_CAST_H
#define PROMETHEUS_CAST_H

#include "kernel.h"

/* Two elements per thread, so launch element_count/2 threads for both. */
unsigned pr_emit_cast_f32_to_f16(hp_word *p);
unsigned pr_emit_cast_f16_to_f32(hp_word *p);

#endif /* PROMETHEUS_CAST_H */
