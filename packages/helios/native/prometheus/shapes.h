/*
 * shapes.h — the constants and shapes every kernel is tested with.
 *
 * WHY THEY ARE SHARED: each of these is read by BOTH the kernel -- through the
 * constant bank or as a codegen parameter -- and the checker that judges its
 * output. A kernel scaled by one number and checked against another passes only
 * by coincidence, so there is exactly one definition of each and both sides
 * read it.
 *
 * WHY THE VALUES ARE WHAT THEY ARE: almost every one is chosen to make a
 * specific wrong implementation fail. The transpose is not square, the slice
 * offset is not zero, the dropout probability is not zero or one, the embedding
 * ids are coprime with their count. Each has its reason recorded next to it,
 * because a constant with no reason is a constant someone will later "simplify"
 * into one that hides a bug.
 */
#ifndef PROMETHEUS_SHAPES_H
#define PROMETHEUS_SHAPES_H

#include "kernel.h"

/* ---- the constants the kernels are fed ---------------------------------- */
/* These are shared deliberately: the table passes them to the GPU through the
 * constant bank and the checker uses the same symbol. A kernel scaled by one
 * number and checked against another would pass only by coincidence. */
#define PR_SCALE_BY 0.25f
#define PR_FILL_VALUE 3.5f
#define PR_CLAMP_LO (-8.0f)
#define PR_CLAMP_HI 20.0f
#define PR_RMS_EPS 1e-5f
#define PR_GELU_K0 0.7978845608028654f
#define PR_GELU_K1 0.044715f
#define PR_SOFTCAP_C 4.0f
#define PR_LOG2_E 1.4426950408889634f
#define PR_LN_2 0.6931471805599453f

/*
 * The matmul test shape: 8x8 times 8x8, which is 64 outputs and therefore
 * exactly PR_N. Square and small on purpose -- a rectangular shape would let a
 * transposed index pass, so the SQUARE case is checked here and the rectangular
 * one separately, where M, N and K are all different and no two can be
 * confused.
 */
/* How many times the loop probe goes round. Not a power of two and not equal
 * to any block or grid dimension, so a trip count confused with a thread index
 * gives a visibly wrong answer rather than an accidentally right one. */
/* The default launch when a kernel does not name its own: 32 threads in each of
 * 2 blocks, which is PR_N elements across more than one block and more than one
 * warp. A single block of 64 would never exercise the block index, and a single
 * warp would hide every intra-warp divergence. */
#define PR_BLOCK 32
#define PR_GRID (PR_N / PR_BLOCK)

#define PR_LOOP_TRIPS 5

/*
 * The indexing shapes. Rectangular on purpose, and rows != cols, so a transpose
 * that returned its input unchanged -- or that swapped the wrong pair of
 * dimensions -- cannot pass. A square shape would let both through.
 */
/* The causal mask is square -- it has to be, it is a token-by-token relation. */
/*
 * AdamW test constants.
 *
 * Deliberately NOT the usual 0.9 / 0.999: those are close enough to one that a
 * kernel which dropped the (1-b) term entirely would still produce nearly the
 * right answer for one step, and one step is all a test runs. At 0.5 and 0.25
 * every term contributes visibly.
 *
 * The kernel is handed 1-b1 and 1-b2 rather than b1 and b2, because it computes
 * m + (1-b1)*(g-m) -- the same arithmetic in fewer instructions, and the
 * subtraction belongs on the host where it happens once.
 */
/*
 * Dropout scale, and the keep pattern.
 *
 * 0.25 is not 1/(1-p) for any p the test uses, and that is fine -- what matters
 * is that it is neither 1 nor 0, so a kernel that ignored the scale entirely
 * and one that zeroed everything both fail. A scale of 1.0 would hide the first.
 */
/*
 * Half-precision conversion, from the IEEE 754 binary16 definition: sign bit,
 * five exponent bits with a bias of 15, ten stored mantissa bits.
 *
 * Implemented here rather than taken from a library, and NOT from the kernel's
 * instructions, because the definition is the oracle. Round-to-nearest-even is
 * the mode F2FP encodes and therefore the mode this must implement -- a
 * truncating reference would disagree on exactly the values that test whether
 * the rounding mode was encoded at all.
 */
NvU32 pr_f32_to_f16_bits(float f);
float pr_f16_bits_to_f32(NvU32 h);
/* The round trip, which is what a cast kernel's output should equal. */
float pr_half_round_trip(float f);

#define PR_DROP_SCALE 0.25f

/*
 * Dropout: a seed and counter that are not round numbers, and p = 0.5.
 *
 * A half is the one probability where a kernel that inverted the comparison
 * would produce a mask with the right COUNT of dropped elements, so the oracle
 * checks each position rather than the total. The seed and counter are odd and
 * unequal because a hash fed zeros can look well-mixed while ignoring an input
 * entirely.
 */
#define PR_DROP_SEED 0x1234567u
#define PR_DROP_COUNTER 0x89abcdu
#define PR_DROP_P 0.5f
/* p * 2^32, the integer threshold the kernel actually compares against. */
#define PR_DROP_THRESHOLD ((NvU32)(PR_DROP_P * 4294967296.0))

/* The same hash the kernel computes, from the murmur3 finalizer definition. */
NvU32 pr_drop_hash(unsigned index);
void pr_fill_ce(volatile NvU32 *logits, volatile NvU32 *targets);
float pr_ce_logit(unsigned r, unsigned c);

/* RMS epsilon for the fused kernel, and the per-feature weight. */
#define PR_RES_EPS 1e-5f

#define PR_ADAM_B1 0.5f
#define PR_ADAM_B2 0.25f
#define PR_ADAM_LR 0.125f
#define PR_ADAM_EPS 0.03125f
#define PR_ADAM_WD 0.0625f

/*
 * Cross-entropy shape: 8 rows of 8 classes, which is PR_N logits.
 *
 * The targets are (3i + 2) mod classes -- never equal to the row index, so a
 * kernel that used the row where it meant the target still produces a number
 * and a wrong one.
 */
#define PR_CE_ROWS 8
#define PR_CE_CLASSES 8
#define PR_CE_TARGET(r) ((3u * (r) + 2u) % PR_CE_CLASSES)

#define PR_MASK_N 8

/* What masked_fill writes where the mask is set. Distinctive and not a value
 * any input takes, so a fill that never fired is visible. */
#define PR_MASK_FILL (-42.0f)

/*
 * Slice: start at 5 and take every third element. Neither is 0 or 1 -- an
 * offset of zero and a stride of one make the slice a copy, which passes with
 * both operands ignored.
 */
#define PR_SLICE_OFFSET 5u
#define PR_SLICE_STRIDE 3u
/* Enough source elements that the last one read stays in bounds. */
#define PR_SLICE_COUNT (PR_N / 4u)

#define PR_TR_ROWS 4
#define PR_TR_COLS 16

/* Embedding: 8 tokens of 8 features, from a table with more rows than tokens so
 * a lookup that ignored the id and used the position would read the wrong row. */
#define PR_EMB_TOKENS 8
#define PR_EMB_DIM 8
/* The table has PR_N entries, so PR_N / PR_EMB_DIM rows. */
#define PR_EMB_ROWS (PR_N / PR_EMB_DIM)

#define PR_MM_M 8
#define PR_MM_N 8
#define PR_MM_K 8


#endif /* PROMETHEUS_SHAPES_H */
