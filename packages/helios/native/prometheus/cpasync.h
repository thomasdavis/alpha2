/*
 * cpasync.h — a hardware validation of cp.async (LDGSTS/LDGDEPBAR/DEPBAR),
 * correctly wired: LDGDEPBAR arms scoreboard SB0 via its write barrier, which is
 * the one control field the first probe lacked (see cpasync.c and the wiring
 * decode in hephaestus/sm86_mem.c).
 *
 * global -> shared -> global as an identity copy where the global->shared leg is
 * performed only by cp.async, reusing pr_fill_ints and chk_copy: a wrong wait
 * reads shared the copy has not filled and chk_copy rejects it.
 *
 * `bytes` is 4 or 16 — the narrow form and the 128-bit form the GEMM staging
 * will use, which has stricter alignment and a different descriptor.
 */
#ifndef HELIOS_PROMETHEUS_CPASYNC_H
#define HELIOS_PROMETHEUS_CPASYNC_H

#include "kernel.h"

unsigned pr_emit_cpasync_copy(hp_word *p, unsigned bytes);

#endif /* HELIOS_PROMETHEUS_CPASYNC_H */
