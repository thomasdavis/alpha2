/*
 * tensor.c — see tensor.h.
 */
#include "tensor.h"

#include <string.h>

/* Generous, because nothing frees intermediates yet: the tape offers a release
 * callback and wiring it naively frees tensors the graph still references, so
 * until that is done properly a step allocates one buffer per operation and a
 * loop needs headroom. The pool still RECYCLES across steps once buffers do get
 * freed; this only bounds how many can be live at once. */
/*
 * The handle is 20 bits of index and 12 of generation, and it used to be 16/16.
 *
 * 16 bits meant the table held 65,536 slots -- and the encoding SILENTLY WRAPPED
 * at the last one. make_handle masks index+1 into the low bits, so slot 65,535
 * produced a handle whose index field was zero and whose generation field was
 * not, which is a non-zero handle that resolves to nothing. The caller's
 * "did the allocation fail" check passes, and the failure surfaces one step
 * later as "allocated handle has no view" -- a message about the view, from a
 * bug in the index.
 *
 * That was reached because nothing frees intermediates, so a long enough run
 * fills the table no matter how large it is. 20 bits buys a million slots and
 * makes the boundary far away; it does not make it go away, which is why
 * exhaustion now returns NONE honestly at the top of the range.
 *
 * 12 bits of generation still distinguishes 4,096 reuses of a slot before a
 * stale handle could alias a live one. That is a real if distant limit, and it
 * is the reason the generation is checked at all.
 */
#define MAX_TENSORS (1u << 20)
#define INDEX_BITS 20
#define INDEX_MASK ((1u << INDEX_BITS) - 1u)
#define GENERATION_MASK ((1u << (32 - INDEX_BITS)) - 1u)

/*
 * Size classes are powers of two from 4 KiB up.
 *
 * Rounding up wastes at most half a buffer and it makes the free list an exact
 * match rather than a search: a request either finds a buffer of its class or
 * allocates one. A best-fit search over arbitrary sizes would reuse memory
 * better and would also mean a 1 MiB request could be served by a 64 MiB
 * buffer, quietly holding sixty-three megabytes hostage for the run.
 *
 * The 4 KiB FLOOR is deliberately kept now that tensors share a slab. A tensor
 * still occupies its whole class, so the slack past its end is its own and a
 * kernel that reads a little past the end sees what it saw before -- untouched
 * bytes, not the next tensor's data. Lowering the floor would have shrunk the
 * guard band and turned an over-read from a harmless zero into a plausible
 * wrong number, which is the failure mode this file is most afraid of.
 */
#define MIN_CLASS_SHIFT 12 /* 4 KiB */
#define NUM_CLASSES 20     /* up to 2 GiB */

/*
 * How much memory one trip to the driver buys.
 *
 * 4 MiB, and it is a MEASURED ceiling rather than a chosen size. gaia_alloc
 * asks for physically CONTIGUOUS pages, and tools/slab_probe.c walks the sizes
 * on this hardware:
 *
 *     256M..8M  alloc FAIL
 *     4M        alloc ok, map_gpu ok, map_host ok
 *
 * which is the kernel's MAX_ORDER limit -- 1024 pages of 4 KiB. 64 MiB was
 * tried first, on arithmetic, and every allocation failed; the pool then served
 * nothing at all and the layer suite went red. A constant a machine silently
 * refuses is worse than a smaller one that works.
 *
 * It is still worth ~1024 carves of the smallest class per trip to the driver,
 * which is the whole point: 283 allocations a step becomes well under one.
 */
#define SLAB_BYTES (4ull * 1024 * 1024)
/* 16 GiB of ceiling. Not a reservation -- slabs are allocated on demand, and
 * this only bounds how many can exist. */
#define MAX_SLABS 4096

/*
 * A slab: one driver allocation, mapped once, carved many times.
 *
 * Bump-only. A slab is never partially reclaimed -- the free list recycles
 * CARVES, not slab space -- so `used` only ever grows. That is what makes the
 * carve a pointer bump with no search and no fragmentation logic.
 */
typedef struct {
  gaia_buffer buf;
  NvU64 used;
} slab;

typedef struct {
  NvU64 offset;    /* where this tensor starts inside its slab */
  NvU64 requested; /* what the caller asked for, not the class size */
  NvU32 generation;
  int slabIndex;
  int inUse;
  int classIndex;
  int nextFree;    /* -1 terminates; links the free list of its class */
  int pendingFree; /* freed, but queued work may still read it */
} slot;

static slot g_slots[MAX_TENSORS];
static slab g_slabs[MAX_SLABS];
static unsigned g_slabCount;
static int g_freeHead[NUM_CLASSES];
/*
 * Buffers released while launches are still queued.
 *
 * A freed buffer cannot go straight back into circulation: a kernel that was
 * enqueued and has not run may still read it, and handing it to the next
 * allocation lets the host overwrite that kernel's input. The result is a
 * finite, plausible, wrong number from an operation that looks unrelated.
 *
 * The synchronous design could not express this -- every free happened after
 * the reader had retired. Under batching it is the difference between a working
 * stack and one that stalls, and the cheap fix (drain on every allocation)
 * removes the batching entirely: it fires once per operation.
 *
 * So: freed buffers wait here, and helios_tensor_retire moves them to the free
 * list when the queue drains. One list walk per flush, not per allocation.
 */
static int g_pendingHead = -1;
static unsigned g_used; /* high-water mark of slots ever created */
static helios_tensor_stats g_stats;
static int g_init;

static void init_once(void) {
  if (g_init) return;
  for (int i = 0; i < NUM_CLASSES; i++) g_freeHead[i] = -1;
  g_init = 1;
}

static int class_of(NvU64 bytes) {
  int c = 0;
  NvU64 size = 1ull << MIN_CLASS_SHIFT;
  while (size < bytes && c < NUM_CLASSES - 1) {
    size <<= 1;
    c++;
  }
  return size < bytes ? -1 : c;
}

/*
 * Resolve a handle, or NULL.
 *
 * The generation check is the whole point. Without it a freed handle indexes a
 * slot that has since been handed to someone else, and writing through it
 * corrupts an unrelated tensor -- with no fault, no error, and no way to trace
 * the damage back to the code that caused it.
 */
static slot *resolve(helios_tensor t) {
  if (t == HELIOS_TENSOR_NONE) return NULL;
  const unsigned index = (t & INDEX_MASK) - 1u;
  if (index >= MAX_TENSORS) return NULL;
  slot *s = &g_slots[index];
  if (!s->inUse) return NULL;
  if ((t >> INDEX_BITS) != (s->generation & GENERATION_MASK)) return NULL;
  return s;
}

/* index+1 must fit the field, so the usable range stops one short of the mask.
 * helios_tensor_alloc refuses beyond that rather than letting it wrap. */
static helios_tensor make_handle(unsigned index, NvU32 generation) {
  return ((index + 1u) & INDEX_MASK) |
         ((generation & GENERATION_MASK) << INDEX_BITS);
}

/*
 * Find room for `size`, adding a slab if the current one is full.
 *
 * Returns the slab index and writes the offset, or -1. A request larger than a
 * standard slab gets a slab of its own rather than a special case elsewhere:
 * one code path, and an oversized tensor still frees and recycles like any
 * other.
 */
static int carve(helios_context *ctx, NvU64 size, NvU64 *offset) {
  if (g_slabCount > 0) {
    slab *s = &g_slabs[g_slabCount - 1];
    if (s->buf.size - s->used >= size) {
      *offset = s->used;
      s->used += size;
      return (int)(g_slabCount - 1);
    }
  }

  if (g_slabCount >= MAX_SLABS) return -1;
  slab *s = &g_slabs[g_slabCount];

  /*
   * Ask for a full slab, then HALVE until the machine agrees.
   *
   * 4 MiB is what this hardware gives (see SLAB_BYTES), but the limit is the
   * kernel's contiguous-page ceiling and that moves with memory pressure and
   * with whatever machine this runs on next. Failing the whole allocation
   * because a slab-sized request was refused would turn a busy machine into a
   * dead backend, when a smaller slab would have served perfectly well.
   *
   * The floor is the request itself: below that the carve cannot be satisfied
   * and -1 is the honest answer.
   */
  NvU64 want = size > SLAB_BYTES ? size : SLAB_BYTES;
  for (;;) {
    memset(s, 0, sizeof *s);
    /* CACHED, not write-combined. A tensor is read by the host constantly --
     * every broadcast, slice, concatenation and permutation walks one, and so
     * does every CPU fallback in autograd -- and a CPU read of write-combined
     * memory bypasses the cache: 161x slower than ordinary memory, measured.
     * The pushbuffer and the QMD keep write-combining, which is the trade they
     * actually want. See gaia_alloc_cached. */
    if (gaia_alloc_cached(&ctx->device, &s->buf, want, GAIA_SYSMEM, 1) == 0 &&
        gaia_map_gpu(&ctx->device, &s->buf) == 0 &&
        gaia_map_host(&ctx->device, &s->buf) == 0)
      break;
    /* Partially constructed is the normal case here -- the allocation may have
     * succeeded and a mapping failed -- and gaia_free is safe on that. */
    gaia_free(&ctx->device, &s->buf);
    if (want <= size) return -1;
    want >>= 1;
    if (want < size) want = size;
  }

  s->buf.size = want;
  s->used = size;
  *offset = 0;
  g_stats.allocations++;
  g_stats.bytesHeld += want;
  return (int)g_slabCount++;
}

helios_tensor helios_tensor_alloc(helios_context *ctx, NvU64 bytes) {
  init_once();
  if (bytes == 0) return HELIOS_TENSOR_NONE;
  const int c = class_of(bytes);
  if (c < 0) return HELIOS_TENSOR_NONE;

  /* A buffer of this class already held? Take it without touching the driver.
   * This is the path that matters: measured at 1.0 us against 802.3 us for a
   * carve that has to ask RM for memory. */
  if (g_freeHead[c] >= 0) {
    const int index = g_freeHead[c];
    slot *s = &g_slots[index];
    g_freeHead[c] = s->nextFree;
    s->nextFree = -1;
    s->inUse = 1;
    s->requested = bytes;
    g_stats.live++;
    g_stats.pooled--;
    return make_handle((unsigned)index, s->generation);
  }

  /* One short of the mask: index+1 is what goes in the field, and letting it
   * wrap is what produced a non-zero handle that resolved to nothing. */
  if (g_used >= MAX_TENSORS - 1u) return HELIOS_TENSOR_NONE;
  slot *s = &g_slots[g_used];
  memset(s, 0, sizeof *s);
  const NvU64 size = 1ull << (MIN_CLASS_SHIFT + c);
  NvU64 offset = 0;
  const int si = carve(ctx, size, &offset);
  if (si < 0) return HELIOS_TENSOR_NONE;

  s->slabIndex = si;
  s->offset = offset;
  s->classIndex = c;
  s->requested = bytes;
  s->generation = 1;
  s->inUse = 1;
  s->nextFree = -1;
  const unsigned index = g_used++;
  g_stats.live++;
  g_stats.carved++;
  return make_handle(index, s->generation);
}

/*
 * A free MARKS the buffer; it does not kill the handle. That happens at the
 * step boundary.
 *
 * Killing it here -- bumping the generation and clearing inUse immediately --
 * is the stricter contract and it is stricter than the tape can honour. Wiring
 * the tape's release callback produces a use-after-free every step: `add` is
 * handed a released [1,4,32,32] attention-scores gradient. Tracing every
 * release of that shape shows no tensor freed twice, so it is one BUFFER with
 * two owners that the tape believes are independent -- a lifetime disagreement
 * between the tape and this backend, not a bug in either alone.
 *
 * The Vulkan backend survives the same tape because it DEFERS: `deferRelease`
 * queues a region and `processPendingDestroys` reclaims it later. This is the
 * same bargain. Between a release and the end of the step the handle still
 * resolves and the memory is untouched, so a premature release is harmless
 * rather than fatal; at the boundary the generation moves, every outstanding
 * handle dies at once, and the memory goes back into circulation.
 *
 * What it gives up: a use-after-free that spans a step boundary is no longer
 * caught, it reads recycled memory. That is the same exposure Vulkan has
 * carried all along, and it buys a pool that actually recycles -- without it
 * nothing is ever reused and a batch-16 step allocates 48 MB it never returns.
 */
void helios_tensor_free(helios_tensor t) {
  slot *s = resolve(t);
  if (!s) return;
  /* Already queued this step. Counting it twice would corrupt the pool's
   * bookkeeping and link the slot into the pending list twice, which makes a
   * cycle and hangs the retire walk. */
  if (s->pendingFree) return;
  const int index = (int)((t & INDEX_MASK) - 1u);
  s->pendingFree = 1;
  s->nextFree = g_pendingHead;
  g_pendingHead = index;
  g_stats.live--;
  g_stats.pooled++;
}

NvU64 helios_tensor_addr(helios_tensor t) {
  const slot *s = resolve(t);
  return s ? g_slabs[s->slabIndex].buf.gpuAddr + s->offset : 0;
}

void *helios_tensor_host(helios_tensor t) {
  slot *s = resolve(t);
  return s ? (void *)((NvU8 *)g_slabs[s->slabIndex].buf.hostPtr + s->offset)
           : NULL;
}

NvU64 helios_tensor_bytes(helios_tensor t) {
  const slot *s = resolve(t);
  return s ? s->requested : 0;
}

void helios_tensor_retire(void) {
  int i = g_pendingHead;
  while (i >= 0) {
    slot *s = &g_slots[i];
    const int next = s->nextFree;
    s->pendingFree = 0;
    /* NOW the handle dies. Bumping the generation kills every outstanding copy
     * of it at once, not just the one that was passed to free. */
    s->generation++;
    s->inUse = 0;
    s->nextFree = g_freeHead[s->classIndex];
    g_freeHead[s->classIndex] = i;
    i = next;
  }
  g_pendingHead = -1;
}

helios_tensor_stats helios_tensor_get_stats(void) { return g_stats; }

void helios_tensor_release_all(helios_context *ctx) {
  /* The SLABS own the memory now, so they are what goes back to the driver.
   * Freeing per slot would hand RM an address it never issued. */
  for (unsigned i = 0; i < g_slabCount; i++)
    gaia_free(&ctx->device, &g_slabs[i].buf);
  memset(g_slots, 0, sizeof g_slots);
  memset(g_slabs, 0, sizeof g_slabs);
  memset(&g_stats, 0, sizeof g_stats);
  g_slabCount = 0;
  g_used = 0;
  g_pendingHead = -1;
  g_init = 0;
  init_once();
}
