/*
 * oracle.c — see oracle.h.
 */
#include "oracle.h"

#include <string.h>

NvU32 pr_f2u(float f) { NvU32 u; memcpy(&u, &f, 4); return u; }
float pr_u2f(NvU32 u) { float f; memcpy(&f, &u, 4); return f; }

/* ---- inputs ------------------------------------------------------------- */

void pr_fill_ints(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++) a[i] = i + 1;
}
/* Strictly positive, so log2 and rsqrt are defined everywhere. */
void pr_fill_pos(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++) a[i] = pr_f2u((float)(i + 1));
}
/* Alternating sign, so relu and negation have something to do. A relu tested
 * only on positive input tests nothing. */
void pr_fill_signed(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++)
    a[i] = pr_f2u((i & 1) ? -(float)(i + 1) : (float)(i + 1));
}

/* Two operands for the binary kernels: a[i] = i+1, b[i] = 2i+3. Distinct, both
 * non-zero so division is defined, and different enough that a kernel which
 * confuses its inputs fails rather than coincidentally agreeing. */
void pr_fill_pair(volatile NvU32 *a, volatile NvU32 *b) {
  for (unsigned i = 0; i < PR_N; i++) {
    a[i] = pr_f2u((float)(i + 1));
    b[i] = pr_f2u((float)(2 * i + 3));
  }
}
float pr_in_a(unsigned i) { return (float)(i + 1); }
float pr_in_b(unsigned i) { return (float)(2 * i + 3); }

float pr_in_pos(unsigned i) { return (float)(i + 1); }
float pr_in_signed(unsigned i) {
  return (i & 1) ? -(float)(i + 1) : (float)(i + 1);
}


static char g_msg[PR_MSG_SIZE];
char *pr_msg(void) { return g_msg; }

/*
 * Token ids for the embedding lookup: table row (5*i + 3) mod PR_EMB_ROWS.
 *
 * Written as raw INTEGERS, not floats -- they are used directly as an index, so
 * a float bit pattern here would address somewhere absurd and the failure would
 * be a fault instead of a wrong answer, which is a worse thing to debug.
 *
 * The stride of 5 against 8 tokens is coprime, so every id is distinct and none
 * equals its own position. A lookup that ignored the id and used the thread's
 * block index would therefore be caught, which the identity mapping would hide.
 */
void pr_fill_embedding(volatile NvU32 *table, volatile NvU32 *ids) {
  for (unsigned i = 0; i < PR_N; i++) table[i] = pr_f2u((float)(i + 1));
  for (unsigned i = 0; i < PR_EMB_TOKENS; i++) ids[i] = pr_emb_id(i);
}

NvU32 pr_emb_id(unsigned i) { return (5u * i + 3u) % PR_EMB_ROWS; }

/*
 * The fill mask: set on every third element.
 *
 * Every third rather than alternate, because an alternating mask is exactly the
 * pattern a warp-level bug would also produce, and because a period that does
 * not divide the warp size puts set and clear lanes in different places in
 * different warps. Both halves are non-empty in every warp, which is what makes
 * the predication actually get exercised.
 */
int pr_mask_set(unsigned i) { return (i % 3u) == 0u; }

void pr_fill_mask(volatile NvU32 *a, volatile NvU32 *mask) {
  for (unsigned i = 0; i < PR_N; i++) {
    a[i] = pr_f2u((float)(i + 1));
    mask[i] = pr_mask_set(i) ? 1u : 0u;
  }
}

/*
 * AdamW inputs.
 *
 * Four tensors, all different, none derivable from another by a constant
 * factor. A gradient equal to the parameter, or a v equal to m squared, would
 * let two of the four be confused with no visible effect.
 *
 * v[0] is exactly ZERO on purpose. That is the state of the second moment on
 * the first step of real training, and it is where a sqrt implemented as
 * reciprocal-of-reciprocal-square-root produces 0 * inf = NaN. The one element
 * most likely to be wrong in practice is therefore the first one tested.
 */
float pr_adam_param(unsigned i) { return (float)(i + 1) * 0.25f; }
float pr_adam_grad(unsigned i) { return (float)(i % 7u) - 3.0f; }
float pr_adam_m(unsigned i) { return (float)(i % 5u) * 0.5f - 1.0f; }
float pr_adam_v(unsigned i) { return i == 0 ? 0.0f : (float)(i % 3u) + 0.5f; }

void pr_seed_adam(volatile NvU32 *param) {
  for (unsigned i = 0; i < PR_N; i++) param[i] = pr_f2u(pr_adam_param(i));
}
void pr_fill_adam(volatile NvU32 *grad, volatile NvU32 *m) {
  for (unsigned i = 0; i < PR_N; i++) {
    grad[i] = pr_f2u(pr_adam_grad(i));
    m[i] = pr_f2u(pr_adam_m(i));
  }
}
void pr_fill_adam_v(volatile NvU32 *v) {
  for (unsigned i = 0; i < PR_N; i++) v[i] = pr_f2u(pr_adam_v(i));
}

/*
 * Inputs for the fused residual kernels.
 *
 * x and the residual are DIFFERENT functions of the index and neither is a
 * multiple of the other, so a kernel that added a tensor to itself, or that
 * used one slot where it meant the other, cannot produce the right answer. The
 * weight is not all-ones for the same reason -- an all-ones weight makes a
 * kernel that ignores the weight indistinguishable from one that applies it.
 *
 * The mask keeps two thirds of the elements. Not a half: an alternating mask is
 * the pattern a lane-indexing bug also produces.
 */
float pr_res_x(unsigned i) { return (float)(i % 9u) - 4.0f; }
float pr_res_r(unsigned i) { return (float)(i % 5u) * 0.5f - 1.0f; }
float pr_res_w(unsigned i) { return 0.5f + (float)(i % 4u) * 0.25f; }
float pr_res_mask(unsigned i) { return (i % 3u) == 0u ? 0.0f : 1.0f; }

void pr_fill_residual(volatile NvU32 *x, volatile NvU32 *res) {
  for (unsigned i = 0; i < PR_N; i++) {
    x[i] = pr_f2u(pr_res_x(i));
    res[i] = pr_f2u(pr_res_r(i));
  }
}
void pr_fill_res_weight(volatile NvU32 *w) {
  for (unsigned i = 0; i < PR_N; i++) w[i] = pr_f2u(pr_res_w(i));
}
void pr_fill_res_mask(volatile NvU32 *m) {
  for (unsigned i = 0; i < PR_N; i++) m[i] = pr_f2u(pr_res_mask(i));
}

/*
 * f32 -> f16, straight from the binary16 definition.
 *
 * The three cases are the three the format has: too large for the exponent
 * range becomes an infinity, too small becomes a subnormal or zero, and
 * everything else shifts the mantissa down by thirteen bits with a
 * round-to-nearest-even at the boundary.
 *
 * The rounding is the part worth reading twice. Adding 0x0fff plus the lowest
 * SURVIVING bit and then shifting is the standard trick: it rounds up when the
 * discarded part is more than half, down when less, and to even when exactly
 * half -- which is what "nearest even" means and what a plain +0x1000 would get
 * wrong precisely at the ties.
 */
NvU32 pr_f32_to_f16_bits(float f) {
  const NvU32 x = pr_f2u(f);
  const NvU32 sign = (x >> 16) & 0x8000u;
  NvS32 exp = (NvS32)((x >> 23) & 0xffu) - 127 + 15;
  const NvU32 mant = x & 0x7fffffu;

  if (((x >> 23) & 0xffu) == 0xffu)              /* inf or NaN */
    return sign | 0x7c00u | (mant ? 0x200u : 0u);
  if (exp >= 0x1f) return sign | 0x7c00u;        /* overflows to infinity */
  if (exp <= 0) {                                /* subnormal, or zero */
    if (exp < -10) return sign;
    const NvU32 full = mant | 0x800000u;
    const unsigned shift = (unsigned)(14 - exp);
    const NvU32 rounded = (full + (1u << (shift - 1))) >> shift;
    return sign | rounded;
  }
  const NvU32 r = mant + 0x0fffu + ((mant >> 13) & 1u);
  if (r & 0x800000u) { exp++; if (exp >= 0x1f) return sign | 0x7c00u; }
  return sign | ((NvU32)exp << 10) | ((r >> 13) & 0x3ffu);
}

float pr_f16_bits_to_f32(NvU32 h) {
  const NvU32 sign = (h & 0x8000u) << 16;
  const NvU32 exp = (h >> 10) & 0x1fu;
  const NvU32 mant = h & 0x3ffu;
  if (exp == 0) {
    if (!mant) return pr_u2f(sign);
    /* Subnormal: normalise by shifting until the implicit bit appears. */
    NvU32 e = 127 - 15 + 1, m = mant;
    while (!(m & 0x400u)) { m <<= 1; e--; }
    return pr_u2f(sign | (e << 23) | ((m & 0x3ffu) << 13));
  }
  if (exp == 0x1f) return pr_u2f(sign | 0x7f800000u | (mant << 13));
  return pr_u2f(sign | ((exp + 127 - 15) << 23) | (mant << 13));
}

float pr_half_round_trip(float f) {
  return pr_f16_bits_to_f32(pr_f32_to_f16_bits(f));
}

/*
 * Cast inputs.
 *
 * Deliberately NOT all exactly representable in f16. Small integers survive the
 * round trip untouched, so a test made only of them would pass with the
 * rounding mode encoded wrongly, or with the conversion replaced by a copy. The
 * eighths are exact; the sevenths are not, and they are what tests the rounding.
 */
float pr_cast_in(unsigned i) {
  return ((float)(i % 11u) - 5.0f) + (float)(i % 7u) / 7.0f;
}

void pr_fill_cast(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++) a[i] = pr_f2u(pr_cast_in(i));
}

/* The same values already packed as half pairs, for the widening direction. */
void pr_fill_packed(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N / 2u; i++)
    a[i] = pr_f32_to_f16_bits(pr_cast_in(2 * i)) |
           (pr_f32_to_f16_bits(pr_cast_in(2 * i + 1)) << 16);
}
