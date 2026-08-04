/*
 * tensor.h — device memory that outlives a single launch.
 *
 * WHAT: handle-based allocation from a pool, with a free list so a buffer
 * released this step is reused next step instead of going back to the driver.
 *
 * WHY A POOL AND NOT gaia_alloc PER TENSOR: allocating through RM is several
 * ioctls plus a GPU page-table update, tens of microseconds each. A training
 * step allocates and frees hundreds of intermediates -- every activation, every
 * gradient -- and the sizes repeat exactly, step after step, forever. Paying
 * the driver for that would cost more than the arithmetic. After the first step
 * the pool should serve every request without a single allocation, and
 * helios_tensor_stats exists so that claim can be checked rather than assumed.
 *
 * WHY HANDLES AND NOT POINTERS: these cross into JavaScript, where a pointer is
 * a number that can be forged, kept past a free, or invented. A handle carries
 * a generation counter, so a stale one is REJECTED rather than silently
 * addressing whatever now lives at that slot -- which is the difference between
 * a clear error and a corrupted tensor with no explanation.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no defragmentation, no sub-allocation
 * within a buffer, no eviction under pressure. Every tensor is its own
 * allocation, rounded up to a size class. That wastes memory on odd sizes and
 * it makes lifetime trivially correct, which is the right trade while
 * correctness is what is being established.
 */
#ifndef HELIOS_TENSOR_H
#define HELIOS_TENSOR_H

#include "context.h"

/*
 * A handle. Zero is always invalid, so a zeroed struct or a forgotten
 * assignment fails loudly instead of addressing slot zero.
 *
 * The low bits index the table and the high bits are the generation, which
 * increments on every free. Both halves are checked on use.
 */
typedef NvU32 helios_tensor;

#define HELIOS_TENSOR_NONE 0u

/* Allocate at least `bytes`, host- and GPU-visible. Returns NONE on failure. */
helios_tensor helios_tensor_alloc(helios_context *ctx, NvU64 bytes);

/* Return it to the pool. The handle is dead afterwards and any further use of
 * it is rejected rather than acted on. */
void helios_tensor_free(helios_tensor t);

/* The GPU address, or 0 if the handle is stale or invalid. */
NvU64 helios_tensor_addr(helios_tensor t);

/* The host mapping, or NULL if the handle is stale or invalid. */
void *helios_tensor_host(helios_tensor t);

/* The usable size in bytes -- what was asked for, not the rounded class. */
NvU64 helios_tensor_bytes(helios_tensor t);

/*
 * How the pool has behaved: how many buffers exist, how many are in use, and
 * how many times a request had to go to the driver.
 *
 * `allocations` is the number worth watching. It should stop growing after the
 * first step; if it does not, something is asking for a size the pool keeps
 * missing, and the cost is invisible except as a step that is mysteriously slow.
 */
typedef struct {
  unsigned live;        /* handed out and not yet freed */
  unsigned pooled;      /* held, available for reuse */
  unsigned allocations; /* trips to the driver, cumulative */
  NvU64 bytesHeld;
} helios_tensor_stats;

helios_tensor_stats helios_tensor_get_stats(void);

/* Release everything back to the driver. For shutdown and for tests. */
void helios_tensor_release_all(helios_context *ctx);

#endif /* HELIOS_TENSOR_H */
