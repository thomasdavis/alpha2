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
 * WHY TENSORS SHARE A SLAB: this file used to say that every tensor being its
 * own allocation was "the right trade while correctness is what is being
 * established". Correctness is established -- the gate is green and the loss
 * matches cpu_ref -- and the trade was then measured: a fresh allocation costs
 * 802.3 us and one served from the free list costs 1.0 us, an 800x gap that is
 * three ioctls and an mmap. Nothing frees intermediates yet, so every one of a
 * step's ~283 allocations paid the 802 us: ~227 ms of a 349 ms step, in the
 * driver, doing no arithmetic. Tensors are now carved from slabs that are
 * mapped once, which makes a first-time allocation a pointer bump.
 *
 * Lifetime is unchanged by this: a carve is still a distinct byte range, still
 * rounded up to a whole size class, still generation-checked, still retired
 * through the pending list on a flush. Only where the bytes come from changed.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no defragmentation, no eviction under
 * pressure, and no reclaim of slab space -- the free list recycles CARVES, so
 * `used` within a slab only grows. A run that never frees will therefore still
 * consume memory at ~1.1 MiB per step; it just no longer pays the driver for
 * it. `carved` is the stat that shows whether that is happening.
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
 * `carved` is the number worth watching, and it is reported SEPARATELY from
 * `allocations` for a reason. Slabs made a trip to the driver rare, so
 * `allocations` alone would have gone quiet on exactly the fault it exists to
 * reveal -- a pool that never recycles reads as healthy when the driver is no
 * longer the thing it costs. `carved` counts requests the free list could not
 * serve, so it still rises once per operation when nothing is being freed.
 *
 * Both should stop growing after the first step. If `carved` does not, either
 * nothing is calling release or something is asking for a size the pool keeps
 * missing, and the cost is invisible except as memory that climbs forever.
 */
typedef struct {
  unsigned live;        /* handed out and not yet freed */
  unsigned pooled;      /* held, available for reuse */
  unsigned allocations; /* trips to the driver (slabs), cumulative */
  unsigned carved;      /* requests the free list could not serve, cumulative */
  NvU64 bytesHeld;
} helios_tensor_stats;

helios_tensor_stats helios_tensor_get_stats(void);

/* Move buffers freed during the last batch into circulation. Called by the
 * context when the queue drains -- until then a freed buffer may still be read
 * by a kernel that has not run. */
void helios_tensor_retire(void);

/* Release everything back to the driver. For shutdown and for tests. */
void helios_tensor_release_all(helios_context *ctx);

#endif /* HELIOS_TENSOR_H */
