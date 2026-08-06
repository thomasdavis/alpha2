/*
 * program.c — see program.h.
 */
#include "program.h"

#include "../prometheus/builders.h"
#include "../prometheus/hmma.h"
#include "../prometheus/normalize.h"
#include "../prometheus/colsum.h"
#include "../prometheus/mask.h"
#include "../prometheus/indexing.h"
#include "../prometheus/qkv_headmajor.h"

#include <string.h>

/*
 * A fixed table with linear probing.
 *
 * Fixed because the shape set is fixed: a model runs the same dozen or so
 * shapes for its whole life, and a table that grows would be machinery serving
 * a case that does not arise. Linear probing because at this occupancy the
 * probe sequence is one or two entries and a chain of pointers would cost more
 * in indirection than it saves in comparisons.
 */
#define CACHE_SLOTS 512

static helios_program g_cache[CACHE_SLOTS];
static unsigned g_count;

static int key_eq(const helios_key *a, const helios_key *b) {
  return a->kind == b->kind && a->arg0 == b->arg0 && a->arg1 == b->arg1 &&
         a->arg2 == b->arg2;
}

/*
 * Mix the four fields rather than adding them.
 *
 * A matmul of 8x16x32 and one of 32x16x8 differ only in the ORDER of their
 * arguments, and any hash that sums or xors them without position collides the
 * two -- returning a program built for the wrong shape, which runs, touches
 * memory it should not, and produces plausible garbage. The multipliers are
 * odd so no field can be shifted out of the result by another.
 */
static unsigned key_hash(const helios_key *k) {
  unsigned h = (unsigned)k->kind * 0x9E3779B1u;
  h = (h ^ k->arg0) * 0x85EBCA77u;
  h = (h ^ k->arg1) * 0xC2B2AE3Du;
  h = (h ^ k->arg2) * 0x27D4EB2Fu;
  return h ^ (h >> 16);
}

/*
 * The block width for a flat elementwise launch: the largest power of two up to
 * 512 that DIVIDES the element count exactly.
 *
 * PR_BLOCK is 32, which is one warp — a size chosen for the 64-element known-
 * answer harness and never revisited for a model. At 32 threads a block, a
 * 327,680-element tensor launches 10,240 blocks, and an SM will hold at most 16
 * of them: 512 of its 1,536 thread slots, a third of the machine, on operations
 * that are 40% of a step's GPU time.
 *
 * EXACT DIVISION, not a ceiling, and that is the whole reason this is a
 * function. A partial last block runs threads past the end of the tensor, and
 * they would not fault — the pool rounds every allocation up to a size class,
 * so a thread reading a little past the end reads that tensor's own slack. It
 * is the WRITE that is the problem, and at the classes this model uses the
 * slack is often zero: 327,680 floats is exactly 1.25 MiB, which is exactly a
 * class. So the launch must cover the tensor and nothing else.
 *
 * The kernel needs no change: it reads ntid from the constant bank, so it has
 * always been block-size agnostic.
 */
static NvU32 ew_block(NvU32 n) {
  for (NvU32 b = 512; b > PR_BLOCK; b >>= 1)
    if (n % b == 0) return b;
  return PR_BLOCK;
}

/* Emit the program for `key` into `p`, returning 0 on success. The launch shape
 * is chosen here because it is chosen WITH the code -- see program.h. */
static int emit(const helios_key *key, helios_program *p) {
  const NvU32 n = key->arg1 ? key->arg1 : PR_N;
  p->sharedBytes = 0;

  switch (key->kind) {
    case HL_ELEMENTWISE:
      p->count = pr_emit_elementwise(p->code, (pr_ew_op)key->arg0);
      p->blockX = ew_block(n);
      p->gridX = (n + p->blockX - 1) / p->blockX;
      return 0;

    case HL_REDUCE:
      /* One block over `n` elements, so the tree fits in shared memory and the
       * barriers are block barriers. A tensor larger than a block is reduced in
       * two passes by the dispatcher, not by this kernel. */
      p->count = pr_emit_reduction(p->code, (pr_red_op)key->arg0, n);
      p->blockX = n;
      p->gridX = 1;
      p->sharedBytes = n * 4;
      return 0;

    case HL_REDUCE_PARTIAL:
      /* arg1 is elements PER BLOCK and arg2 the block count, so the launch
       * covers arg1*arg2 elements and writes arg2 partial results. */
      p->count = pr_emit_reduction_partial(p->code, (pr_combine)key->arg0,
                                           key->arg1);
      p->blockX = key->arg1;
      p->gridX = key->arg2;
      p->sharedBytes = key->arg1 * 4;
      return 0;

    case HL_NORMALIZE:
      /* arg1 is the row WIDTH and arg2 the row count: one block per row, each
       * with its own shared memory, so rows normalise independently. */
      p->count = pr_emit_normalize(p->code, (pr_norm_op)key->arg0, n);
      /* Softmax over a row wider than a block walks it in chunks, and then only
       * the per-thread partials go through shared memory — 4 KB rather than the
       * 48 KB a 12,288-class row would have asked for. */
      p->blockX = pr_normalize_block((pr_norm_op)key->arg0, n);
      p->gridX = key->arg2 ? key->arg2 : 1;
      p->sharedBytes = pr_normalize_block((pr_norm_op)key->arg0, n) * 4;
      return 0;

    case HL_COLUMN_SUM:
      /* arg0 = rows, arg1 = cols. One block per 32 columns, 32 row-lanes deep;
       * the row axis is walked, not split, so the grid is in columns alone. */
      /* arg2 selects the PRODUCT form, which takes a second input. Part of
       * the key: the two emit different programs over different parameter
       * counts, and a cache hit across them would read a parameter slot the
       * launch never filled. */
      p->count = pr_emit_column_sum(p->code, key->arg0, key->arg1, (int)key->arg2);
      p->blockX = PR_COLSUM_BLOCK;
      p->gridX = pr_colsum_grid(key->arg1);
      p->sharedBytes = pr_colsum_shared();
      return 0;

    case HL_MATMUL_T_CP:
      /*
       * The cp.async f16 GEMM (NT). Requires the same shape divisibility as the
       * staged path — it reuses pr_hmma_applies, pr_hmma_threads, the block
       * geometry and the register count — but its operands are f16 and it stages
       * with cp.async. Falls through to the scalar fallback below only if the
       * shape does not divide, exactly like the staged path.
       */
      if (pr_hmma_applies(key->arg0, key->arg1, key->arg2)) {
        p->count = pr_emit_hmma_cpasync(p->code, key->arg0, key->arg1, key->arg2, 0);
        p->blockX = pr_hmma_threads();
        p->gridX = key->arg0 / pr_hmma_block_rows();
        p->gridY = key->arg1 / pr_hmma_block_cols();
        p->sharedBytes = pr_hmma_shared();
        p->regs = pr_hmma_regs();
        return 0;
      }
      return -1; /* no scalar f16 fallback — the caller keeps the f32 path */

    case HL_MATMUL:
    case HL_MATMUL_T:
    case HL_MATMUL_TA:
    case HL_MATMUL_ACC:
    case HL_MATMUL_T_ACC:
    case HL_MATMUL_TA_ACC:
      /*
       * The TENSOR-CORE path, when the shape divides.
       *
       * Chosen here rather than by the caller so that one function decides both
       * the code and the launch that must match it. The two disagreeing is not
       * a wrong answer — it is an invalid launch, and those fault the channel
       * asynchronously at whatever flushes next.
       */
      if (pr_hmma_applies(key->arg0, key->arg1, key->arg2)) {
        const int acc = key->kind == HL_MATMUL_ACC ||
                        key->kind == HL_MATMUL_T_ACC ||
                        key->kind == HL_MATMUL_TA_ACC;
        const pr_mm_kind layout =
            (key->kind == HL_MATMUL_T || key->kind == HL_MATMUL_T_ACC) ? PR_MM_NT
            : (key->kind == HL_MATMUL_TA || key->kind == HL_MATMUL_TA_ACC) ? PR_MM_TA
                                                                           : PR_MM_NN;
        p->count = pr_emit_hmma(p->code, key->arg0, key->arg1, key->arg2,
                                layout, acc);
        p->blockX = pr_hmma_threads();
        p->gridX = key->arg0 / pr_hmma_block_rows();
        p->gridY = key->arg1 / pr_hmma_block_cols();
        p->sharedBytes = pr_hmma_shared();
        p->regs = pr_hmma_regs();
        return 0;
      }
      /* The scalar kernel has no transposed-A form. hl_matmul_transposed_a
       * refuses the call when the tensor path does not apply, and the caller
       * materialises a transpose as it always did — so this is unreachable
       * rather than wrong, and returning an error beats emitting an NN kernel
       * for a TA key. */
      /* The scalar kernel has neither a transposed-A form nor an accumulating
       * one; both callers refuse when the tensor path does not apply. */
      if (key->kind == HL_MATMUL_TA || key->kind == HL_MATMUL_ACC ||
          key->kind == HL_MATMUL_T_ACC || key->kind == HL_MATMUL_TA_ACC)
        return -1;
      p->count = pr_emit_matmul_kind(p->code, key->arg0, key->arg1, key->arg2,
                                     key->kind == HL_MATMUL_T);
      p->sharedBytes = pr_matmul_colblocked(key->arg0, key->arg1, key->arg2)
                           ? key->arg2 * 4u
                       : pr_matmul_tiled(key->arg0, key->arg1, key->arg2)
                           ? pr_matmul_tiled_shared(key->arg0, key->arg1, key->arg2)
                           : pr_matmul_shared_bytes(key->arg1, key->arg2);
      /* At most 1024 threads: a thread walks its column in chunks of the block
       * width, so N is no longer bounded by the hardware's block limit. Asking
       * for more was an invalid launch, which faults the channel (0xd) rather
       * than failing the call. */
      p->blockX = pr_matmul_colblocked(key->arg0, key->arg1, key->arg2)
                      ? 256u : pr_matmul_block(key->arg1);
      p->gridX = key->arg0 / pr_matmul_rows(key->arg0, key->arg1, key->arg2);
      /* Zero = the default. The two-row tiled map is packed under 44 so it
       * does not need more, and asking for more is not free: at blockX 640,
       * 64 registers fits one block an SM where 48 fits two. This is the hook
       * a four-row tile (highest register R53) will use. */
      p->regs = pr_matmul_tiled(key->arg0, key->arg1, key->arg2) ? 64u : 0u;
      /* The batch rides in the Y grid. It is not part of the KEY: the emitted
       * code is identical for any batch -- the plane strides come from M, N and
       * K -- so keying on it would generate one program per batch size for no
       * difference. */
      return 0;

    case HL_SLICE_ROWS:
      p->count = pr_emit_slice_rows(p->code, key->arg0, key->arg1);
      p->blockX = pr_row_block(key->arg0);
      p->gridX = 1;
      p->gridY = 0; /* rows supplied at launch */
      return 0;

    case HL_BROADCAST:
      p->count = pr_emit_broadcast(p->code, key->arg0, key->arg1);
      p->blockX = pr_row_block(key->arg1);
      p->gridX = 1;
      p->gridY = 0;
      return 0;

    case HL_CAT_ROWS:
      p->count = pr_emit_cat_rows(p->code, key->arg0, key->arg1);
      p->blockX = pr_row_block(key->arg0);
      p->gridX = 1;
      p->gridY = 0;
      return 0;

    case HL_PERMUTE:
      /* h on X, t on Y, b on Z, one thread per feature. Every index is a grid
       * index, which is what frees the kernel from needing any dimension to be
       * a power of two — see pr_emit_permute. The caller supplies the batch as
       * the third dimension at launch. */
      p->count = pr_emit_permute(p->code, key->arg0, key->arg1, key->arg2);
      /* A block covers R t-values, so it is R*D threads and there are T/R of
       * them down the Y axis. R divides T by construction — a partial block
       * would read and write past the plane. */
      {
        const NvU32 R = pr_permute_rows(key->arg0, key->arg2);
        p->blockX = key->arg2 * R; /* D * R */
        p->gridX = key->arg1;      /* H */
        p->gridY = key->arg0 / R;  /* T / R */
      }
      return 0;

    case HL_SLICE_QKV_HM:
    case HL_SLICE_QKV_HM_BWD:
      /* Same launch shape as permute — h on X, t on Y, b on Z — but the source
       * is a grouped qkvFlat row of width 3H and the plane (q/k/v) is baked in.
       * arg1 packs the plane in its high half so one shape's three planes are
       * three cached programs. The backward kind swaps load and store. */
      {
        const unsigned H = key->arg1 & 0xFFFFu;
        const unsigned plane = key->arg1 >> 16;
        const int backward = (key->kind == HL_SLICE_QKV_HM_BWD);
        const NvU32 R = pr_permute_rows(key->arg0, key->arg2);
        p->count = pr_emit_qkv_headmajor(p->code, key->arg0, H, key->arg2,
                                         plane, backward);
        p->blockX = key->arg2 * R; /* D * R */
        p->gridX = H;              /* H */
        p->gridY = key->arg0 / R;  /* T / R */
      }
      return 0;

    case HL_TRANSPOSE:
      /* arg2 is the batch: one block per (row, plane), so the whole batch is
       * one launch instead of one per plane with a host copy between. */
      p->count = pr_emit_transpose(p->code, key->arg0, key->arg1);
      /* At most 1024: threads walk the columns in chunks, so a transpose of a
       * 12,288-wide weight is the same launch as one of a 64-wide activation. */
      p->blockX = pr_transpose_block(key->arg1);
      p->gridX = key->arg0;
      p->gridY = key->arg2 ? key->arg2 : 1;
      return 0;

    case HL_EMBEDDING:
      p->count = pr_emit_embedding(p->code, key->arg0);
      p->blockX = key->arg0; /* one thread per feature */
      p->gridX = key->arg1;  /* one block per token */
      return 0;

    case HL_SLICE:
      p->count = pr_emit_slice(p->code);
      p->blockX = key->arg0 < PR_BLOCK ? key->arg0 : ew_block(key->arg0);
      p->gridX = (key->arg0 + p->blockX - 1) / p->blockX;
      return 0;

    case HL_CAUSAL_MASK:
      p->count = pr_emit_causal_mask(p->code, key->arg0);
      p->blockX = key->arg0;
      p->gridX = key->arg1;
      return 0;

    case HL_MASKED_FILL:
      /* arg0 is the mask's element count when the mask is TILED, and 0 when it
       * has already been materialised to the value's size. Part of the key, so
       * the two forms cannot share a cached program. */
      p->count = key->arg0 ? pr_emit_masked_fill_tiled(p->code, key->arg0)
                           : pr_emit_masked_fill(p->code);
      p->blockX = ew_block(n);
      p->gridX = (n + p->blockX - 1) / p->blockX;
      return 0;

    case HL_CAST_TO_F16:
    case HL_CAST_TO_F32:
      /* A pair per thread, so half the threads and an even element count. */
      p->count = key->kind == HL_CAST_TO_F16
                     ? pr_emit_cast_f32_to_f16(p->code)
                     : pr_emit_cast_f16_to_f32(p->code);
      /* Half the threads, so the DIVISOR is over the pair count. */
      p->blockX = ew_block(n / 2) ;
      p->gridX = (n / 2 + p->blockX - 1) / p->blockX;
      return 0;

    case HL_DROPOUT:
      p->count = pr_emit_dropout_mask(p->code);
      p->blockX = ew_block(n);
      p->gridX = (n + p->blockX - 1) / p->blockX;
      return 0;

    case HL_CROSS_ENTROPY:
      p->count = pr_emit_cross_entropy(p->code, key->arg0);
      /* At most 1024, and shared memory holds the per-THREAD partials rather
       * than the whole row — 4 KB whatever the vocabulary is. */
      p->blockX = pr_cross_entropy_block(key->arg0);
      p->gridX = key->arg1;  /* one block per row */
      p->sharedBytes = pr_cross_entropy_block(key->arg0) * 4;
      return 0;

    case HL_EMBEDDING_SCATTER:
      /* One block per token, one thread per column — the forward's geometry,
       * which is why `dim` is capped at a block there and here. */
      p->count = pr_emit_embedding_scatter(p->code, key->arg0);
      p->blockX = key->arg0;
      p->gridX = key->arg1;
      return 0;

    case HL_CROSS_ENTROPY_BACKWARD:
      /* Same block shape as the forward: one block per row, shared memory
       * holding per-thread partials rather than the row. */
      p->count = pr_emit_cross_entropy_backward(p->code, key->arg0);
      p->blockX = pr_cross_entropy_block(key->arg0);
      p->gridX = key->arg1;
      p->sharedBytes = pr_cross_entropy_block(key->arg0) * 4;
      return 0;

    case HL_RESIDUAL_RMS:
      p->count = pr_emit_residual_rms(p->code, key->arg0);
      p->blockX = key->arg0;
      p->gridX = key->arg1 ? key->arg1 : 1;
      p->sharedBytes = key->arg0 * 4;
      return 0;

    case HL_RESIDUAL_DROPOUT:
      p->count = pr_emit_residual_dropout(p->code);
      p->blockX = key->arg0;
      p->gridX = key->arg1 ? key->arg1 : 1;
      return 0;

    case HL_ADAMW:
      p->count = pr_emit_adamw(p->code);
      p->blockX = PR_BLOCK;
      p->gridX = (n + PR_BLOCK - 1) / PR_BLOCK;
      return 0;

    case HL_KIND_COUNT:
      break;
  }
  return -1;
}

#define HELIOS_DEFAULT_REGS 48u
static NvU32 g_lastRegs = HELIOS_DEFAULT_REGS;

const helios_program *helios_program_get(helios_key key) {
  unsigned slot = key_hash(&key) % CACHE_SLOTS;
  for (unsigned probe = 0; probe < CACHE_SLOTS; probe++) {
    helios_program *p = &g_cache[(slot + probe) % CACHE_SLOTS];
    if (p->used) {
      if (key_eq(&p->key, &key)) { g_lastRegs = p->regs; return p; }
      continue;
    }
    memset(p, 0, sizeof *p);
    p->key = key;
    if (emit(&key, p) != 0) return NULL;
    /* An emitter that overran its buffer has already corrupted the next entry,
     * so this cannot catch it -- what it catches is one that returned nothing,
     * which is an unimplemented kind reaching here. */
    if (p->count == 0 || p->count > PR_MAX_INSTRUCTIONS) return NULL;
    if (p->regs == 0) p->regs = HELIOS_DEFAULT_REGS;
    p->used = 1;
    g_count++;
    g_lastRegs = p->regs;
    return p;
  }
  return NULL;
}

NvU32 helios_program_last_regs(void) { return g_lastRegs; }

unsigned helios_program_count(void) { return g_count; }

void helios_program_reset(void) {
  memset(g_cache, 0, sizeof g_cache);
  g_count = 0;
}
