/*
 * qkv_headmajor.h — see qkv_headmajor.c.
 */
#ifndef PROMETHEUS_QKV_HEADMAJOR_H
#define PROMETHEUS_QKV_HEADMAJOR_H

#include "kernel.h"

/*
 * One plane of a grouped token-major QKV projection to head-major, in one pass.
 *
 * Forward (backward=0) gathers qkvFlat [B*T, 3*H*D] into a head-major output
 * [B, H, T, D]: out[b][h][t][d] = qkvFlat[b*T+t][plane*H*D + h*D + d]. Backward
 * (backward=1) scatters a head-major gradient back to the plane's columns of a
 * qkvFlat-shaped gradient. plane selects q (0), k (1) or v (2).
 *
 * Launch is permute's: grid (H, T/pr_permute_rows(T,D), batch) with
 * pr_permute_rows(T,D)*D threads a block. D must be a power of two.
 */
unsigned pr_emit_qkv_headmajor(hp_word *p, unsigned T, unsigned H, unsigned D,
                               unsigned plane, int backward);

#endif /* PROMETHEUS_QKV_HEADMAJOR_H */
