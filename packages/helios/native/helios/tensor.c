/*
 * tensor.c — see tensor.h.
 */
#include "tensor.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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
/*
 * QUARTER-OCTAVE size classes: 1.00, 1.25, 1.50 and 1.75 times each power of
 * two, rather than powers of two alone.
 *
 * The classes exist so a freed buffer can be handed to the next request without
 * asking the driver — 1.0 us against 802.3 — and rounding to a power of two is
 * the cheapest way to make that work. It is also, at this model's shapes, the
 * most expensive:
 *
 *     activation   512 x  640 x 4   1.25 MiB -> 2 MiB    60% wasted
 *     mlp          512 x 2560 x 4   5.00     -> 8        60%
 *     logits       512 x12288 x 4  24.00     ->32        33%
 *
 * A 105M model at batch 8 held 6.2 GB of an 8 GB card against a working set
 * near 2.5, and the waste was the binding constraint on everything: batch 12
 * would not run, the fused layerNorm backward could not be measured because it
 * could not allocate, and the GEMM wants more rows than the memory allows.
 *
 * Quarters cut the worst case from 100% to 25% and, because these shapes are
 * multiples of 640 rather than arbitrary, take the three above to ZERO — 1.25,
 * 5.00 and 24.00 MiB are each exactly a class. The cost is four times as many
 * free lists, which is four times as many pointers.
 *
 * The 4 KiB FLOOR and its guard-band reasoning above are unchanged: the
 * smallest class is still 4 KiB and a tensor still owns its whole class.
 */
#define MIN_CLASS_SHIFT 12 /* 4 KiB */
#define CLASS_SUBS 4       /* steps within an octave */
#define NUM_OCTAVES 20     /* up to 2 GiB */
#define NUM_CLASSES (NUM_OCTAVES * CLASS_SUBS)

/* (4 + sub) << (shift - 2 + octave): sub 0 is the octave itself, so class 0 is
 * exactly the 4 KiB floor and every octave boundary lands on a power of two. */
static NvU64 class_size(int c) {
  return (NvU64)(CLASS_SUBS + (c % CLASS_SUBS))
         << (MIN_CLASS_SHIFT - 2 + (c / CLASS_SUBS));
}

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
  /* Backed by several chunks in one VA range rather than one allocation, so it
   * must be released through gaia_free_large. See the large-allocation branch
   * in new_slab. */
  int large;
  /*
   * Whether this slab is mapped into the host's address space.
   *
   * It is a property of the SLAB and not of the tensor because the mapping is
   * made once, for the whole slab, at the one trip to the driver. A tensor is a
   * byte range inside it and inherits whatever the slab has.
   */
  int hostVisible;
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
  int hostVisible; /* copied from the slab, so a free can find its list */
} slot;

static slot g_slots[MAX_TENSORS];
static slab g_slabs[MAX_SLABS];
static unsigned g_slabCount;
/*
 * TWO POOLS, and the free list has to know which one a buffer came from.
 *
 * Video memory is the whole point -- the GPU reads system memory at 19.7 GB/s
 * against 111.8 measured from its own -- but it cannot simply replace system
 * memory, because video memory is reachable from the host only through the BAR1
 * aperture, and this card's is 256 MiB against a batch-128 step's 1.4 GB. So
 * the two kinds of memory are not interchangeable and a single free list would
 * hand a device-resident buffer to a caller that needs to read it on the host,
 * which is not slow, it is a null pointer.
 *
 * Hence: index [class][hostVisible]. The default pool answers helios_tensor_alloc
 * and holds the tensors kernels work on; the host-visible pool answers
 * helios_tensor_alloc_host and holds the staging buffers through which the host
 * reads them. With HELIOS_VIDMEM unset both pools are system memory and this
 * whole distinction costs one branch, which is what keeps the default path
 * exactly as it was.
 */
static int g_freeHead[NUM_CLASSES][2];
/* The slab each pool is currently carving from, or -1. */
static int g_current[2] = {-1, -1};
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

static helios_zero_fn g_zeroFn;

void helios_tensor_set_zero_fn(helios_zero_fn fn) { g_zeroFn = fn; }

static void init_once(void) {
  if (g_init) return;
  for (int i = 0; i < NUM_CLASSES; i++) g_freeHead[i][0] = g_freeHead[i][1] = -1;
  g_init = 1;
}

/*
 * Video memory only when asked, and read ONCE.
 *
 * getenv in the allocation path would be a syscall-free but still repeated
 * string walk on the hottest function in the file; more to the point, a value
 * that changed halfway through a run would split the pools against themselves.
 */
static int vidmem_enabled(void) {
  static int cached = -1;
  if (cached < 0) cached = getenv("HELIOS_VIDMEM") ? 1 : 0;
  return cached;
}

static int class_of(NvU64 bytes) {
  int c = 0;
  while (class_size(c) < bytes && c < NUM_CLASSES - 1) c++;
  return class_size(c) < bytes ? -1 : c;
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
static int carve(helios_context *ctx, NvU64 size, NvU64 *offset, int hostVisible) {
  /*
   * The open slab is now PER POOL.
   *
   * This looked at the last slab created, which was right when there was one
   * kind. With two, a single staging allocation between two device allocations
   * would close the device slab at whatever it had reached and start another --
   * the space is not lost, but the pools would interleave and each would carve
   * from a slab the other had just been filling.
   */
  const int p = hostVisible ? 1 : 0;
  /*
   * A LARGE REQUEST GETS ITS OWN SLAB, SIZED TO IT.
   *
   * The bump allocator keeps one open slab per pool and abandons its remainder
   * the moment a request does not fit. With power-of-two classes that never
   * happened — every class divided 4 MiB exactly, so a slab packed perfectly —
   * and with quarter-octave classes it happens constantly: a 2.5 MiB carve
   * takes a 4 MiB slab and strands 1.5 MiB, which is worse waste than the
   * rounding the finer classes were introduced to remove.
   *
   * Anything over half a slab cannot share with another of its own size
   * anyway, so it takes a dedicated allocation of exactly its class and strands
   * nothing. Smaller carves keep packing, where a tail is at most a quarter of
   * what the class already rounded to.
   */
  const int dedicated = size > SLAB_BYTES / 2;
  if (!dedicated && g_current[p] >= 0) {
    slab *s = &g_slabs[g_current[p]];
    if (s->buf.size - s->used >= size) {
      *offset = s->used;
      s->used += size;
      return g_current[p];
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
  NvU64 want = dedicated ? size : SLAB_BYTES;
  for (;;) {
    memset(s, 0, sizeof *s);
    /* CACHED, not write-combined. A tensor is read by the host constantly --
     * every broadcast, slice, concatenation and permutation walks one, and so
     * does every CPU fallback in autograd -- and a CPU read of write-combined
     * memory bypasses the cache: 161x slower than ordinary memory, measured.
     * The pushbuffer and the QMD keep write-combining, which is the trade they
     * actually want. See gaia_alloc_cached. */
    /*
     * VIDMEM for the device pool, and NO HOST MAPPING for it.
     *
     * The first attempt at video memory kept the host mapping and cost 60x. The
     * mapping was the whole of it: video memory reaches the host through the
     * BAR1 aperture, 256 MiB on this card, uncached and a PCIe round trip per
     * access -- so every host read in the stack moved onto the slowest path
     * available, and a 4 MiB slab could not even be allocated once the aperture
     * filled. Not mapping it removes both: the aperture is untouched, so all 8
     * GiB is usable, and there is no uncached pointer for anything to read
     * through by accident.
     *
     * What the host needs instead is a STAGING buffer in system memory and a
     * copy kernel, which is the host-visible pool below. That turns a host read
     * from millions of uncached per-element round trips into one sequential
     * device-to-system copy.
     *
     * With HELIOS_VIDMEM unset, `where` is SYSMEM and hostVisible is forced on
     * for both pools, which is byte-for-byte the previous behaviour.
     */
    const int vid = vidmem_enabled() && !hostVisible;
    const gaia_location where = vid ? GAIA_VIDMEM : GAIA_SYSMEM;
    const int wantHost = !vid;
    /*
     * CACHED only when there will be a host mapping to cache.
     *
     * The coherency attribute describes how the CPU sees the pages, which is a
     * question about system memory. Asking RM for CACHED video memory is asking
     * it to cache an aperture, and it refuses: a 4 MiB VIDMEM slab failed to
     * allocate outright, the loop halved it down to something that worked, and
     * the model then died in layerNorm on a slab too small to carve from. The
     * error surfaced three layers from its cause, which is what an allocator
     * that silently accepts less than it was asked for buys you.
     */
    /*
     * WHICH STAGE REFUSED, not merely that something did.
     *
     * The halving loop turns any refusal into a smaller slab, and a slab
     * smaller than the request turns into -1 several frames later --
     * "layerNorm failed on the device", which names neither the size nor the
     * stage. Three stages can fail here and they want different fixes:
     * allocation is RM saying no to the size or the attributes, map_gpu is the
     * address space, map_host is the aperture.
     */
    const int rcAlloc = gaia_alloc_cached(&ctx->device, &s->buf, want, where, wantHost);
    const int rcGpu = rcAlloc == 0 ? gaia_map_gpu(&ctx->device, &s->buf) : -1;
    const int rcHost = (rcGpu == 0 && wantHost) ? gaia_map_host(&ctx->device, &s->buf) : 0;
    if (rcAlloc == 0 && rcGpu == 0 && rcHost == 0) {
      s->hostVisible = wantHost;
      break;
    }
    if (getenv("HELIOS_TRACE_ALLOC"))
      fprintf(stderr, "[helios] slab %llu KiB %s: alloc=%d map_gpu=%d map_host=%d\n",
              (unsigned long long)(want / 1024), where == GAIA_VIDMEM ? "vidmem" : "sysmem",
              rcAlloc, rcGpu, rcHost);
    /* Partially constructed is the normal case here -- the allocation may have
     * succeeded and a mapping failed -- and gaia_free is safe on that. */
    gaia_free(&ctx->device, &s->buf);
    if (want <= size) {
      /*
       * ONE CONTIGUOUS BLOCK IS NOT THE ONLY WAY TO GET THE BYTES, and this is
       * where that used to be assumed.
       *
       * The halving loop's floor is the request itself, so a dedicated carve
       * larger than the kernel's MAX_ORDER ceiling reached here and returned -1
       * — "allocation of 1146880 floats failed". That reads as a full card and
       * it is not: the failing request was 4.59 MB on an 8 GiB card holding
       * 4.75. gaia_alloc asks RM for physically CONTIGUOUS pages, and 4 MiB is
       * all the kernel will give in one piece whatever else is free.
       *
       * So the batch this stack could run was capped at 24 by the size of a
       * single activation, [24,64,640] being 3.93 MB and [28,64,640] being
       * 4.59, and it looked like a memory-capacity limit for as long as nobody
       * subtracted.
       *
       * gaia_alloc_large reserves one VA range and places several chunks at
       * consecutive addresses inside it. The GPU sees one contiguous buffer
       * because its MMU says so. Video memory only, and dedicated only: a
       * shared slab is 4 MiB by construction and never needs this.
       */
      if (!dedicated || where != GAIA_VIDMEM) return -1;
      if (gaia_alloc_large(&ctx->device, &s->buf, size, SLAB_BYTES) != 0) {
        if (getenv("HELIOS_TRACE_ALLOC"))
          fprintf(stderr, "[helios] large alloc %llu KiB vidmem failed\n",
                  (unsigned long long)(size / 1024));
        return -1;
      }
      s->hostVisible = 0;
      s->large = 1;
      want = size;
      break;
    }
    want >>= 1;
    if (want < size) want = size;
  }

  s->buf.size = want;
  s->used = size;
  *offset = 0;
  g_stats.allocations++;
  g_stats.bytesHeld += want;
  /* A dedicated slab is FULL, so it must not become the open one — leaving it
   * open would close whatever was being packed and strand that remainder, the
   * very waste this branch exists to avoid. */
  if (!dedicated) g_current[p] = (int)g_slabCount;
  return (int)g_slabCount++;
}

static helios_tensor alloc_from(helios_context *ctx, NvU64 bytes, int hostVisible) {
  init_once();
  if (bytes == 0) return HELIOS_TENSOR_NONE;
  const int c = class_of(bytes);
  if (c < 0) return HELIOS_TENSOR_NONE;
  /* Without video memory the pools are the same memory, so collapsing them
   * keeps one free list warm instead of splitting the pool in half. */
  const int p = (hostVisible && vidmem_enabled()) ? 1 : 0;

  /* A buffer of this class already held? Take it without touching the driver.
   * This is the path that matters: measured at 1.0 us against 802.3 us for a
   * carve that has to ask RM for memory. */
  if (g_freeHead[c][p] >= 0) {
    const int index = g_freeHead[c][p];
    slot *s = &g_slots[index];
    g_freeHead[c][p] = s->nextFree;
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
  const NvU64 size = class_size(c);
  NvU64 offset = 0;
  const int si = carve(ctx, size, &offset, p);
  if (si < 0) return HELIOS_TENSOR_NONE;

  s->slabIndex = si;
  s->offset = offset;
  s->classIndex = c;
  s->requested = bytes;
  s->generation = 1;
  s->inUse = 1;
  s->nextFree = -1;
  s->hostVisible = p;
  const unsigned index = g_used++;
  g_stats.live++;
  g_stats.carved++;
  const helios_tensor t = make_handle(index, s->generation);
  /*
   * A fresh carve out of video memory is not zero, and callers assume it is.
   *
   * Only on a CARVE: a buffer served from the free list holds the last tensor's
   * bytes, which is exactly what system memory does too, so the two paths agree
   * and the pool stays a pointer bump. Carves stop after the first step or two,
   * so this is a few thousand fills once and nothing thereafter.
   */
  if (g_zeroFn && !g_slabs[si].hostVisible) g_zeroFn(ctx, t, size);
  return t;
}

helios_tensor helios_tensor_alloc(helios_context *ctx, NvU64 bytes) {
  return alloc_from(ctx, bytes, 0);
}

helios_tensor helios_tensor_alloc_host(helios_context *ctx, NvU64 bytes) {
  return alloc_from(ctx, bytes, 1);
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
  /* A device-resident tensor has no host mapping, and saying so is the point:
   * the alternative is arithmetic on a NULL slab pointer, which produces a
   * plausible address that faults somewhere else entirely. Callers that need
   * the bytes on the host allocate a staging tensor and copy. */
  if (!s || !g_slabs[s->slabIndex].hostVisible) return NULL;
  return (void *)((NvU8 *)g_slabs[s->slabIndex].buf.hostPtr + s->offset);
}

int helios_tensor_host_visible(helios_tensor t) {
  const slot *s = resolve(t);
  return s ? g_slabs[s->slabIndex].hostVisible : 0;
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
    s->nextFree = g_freeHead[s->classIndex][s->hostVisible];
    g_freeHead[s->classIndex][s->hostVisible] = i;
    i = next;
  }
  g_pendingHead = -1;
}

helios_tensor_stats helios_tensor_get_stats(void) { return g_stats; }

/*
 * How many LIVE slots there are of each size class, and how many bytes.
 *
 * The JavaScript census names the allocation SITE of every buffer it did not
 * see released — which is not the same as naming what the pool is still
 * holding, and reading it as if it were sent an afternoon at the wrong
 * function. Removing 84 MB of fallback transposes it had fingered moved the
 * leak by exactly zero.
 *
 * This asks the allocator instead. A class is a power of two, so the histogram
 * IDENTIFIES the tensors: at 18 layers, 640 embd, vocab 12,288 there is exactly
 * one shape per class that the model allocates in quantity.
 */
_Static_assert(NUM_CLASSES <= HELIOS_TENSOR_MAX_CLASSES,
               "callers size their histograms with HELIOS_TENSOR_MAX_CLASSES");

unsigned helios_tensor_class_count(void) { return NUM_CLASSES; }

unsigned long long helios_tensor_class_size(unsigned c) {
  return c < NUM_CLASSES ? (unsigned long long)class_size((int)c) : 0ull;
}

void helios_tensor_live_by_class(unsigned *counts, NvU64 *bytes) {
  for (int c = 0; c < NUM_CLASSES; c++) { counts[c] = 0; bytes[c] = 0; }
  for (unsigned i = 0; i < g_used; i++) {
    const slot *s = &g_slots[i];
    if (!s->inUse || s->pendingFree) continue;
    counts[s->classIndex]++;
    bytes[s->classIndex] += class_size(s->classIndex);
  }
}

void helios_tensor_release_all(helios_context *ctx) {
  /* The SLABS own the memory now, so they are what goes back to the driver.
   * Freeing per slot would hand RM an address it never issued.
   *
   * A large slab holds several chunks in one VA range and only the first is
   * `handle`; gaia_free would release that one and leak the rest along with
   * their mappings. */
  for (unsigned i = 0; i < g_slabCount; i++) {
    if (g_slabs[i].large) gaia_free_large(&ctx->device, &g_slabs[i].buf);
    else gaia_free(&ctx->device, &g_slabs[i].buf);
  }
  memset(g_slots, 0, sizeof g_slots);
  memset(g_slabs, 0, sizeof g_slabs);
  memset(&g_stats, 0, sizeof g_stats);
  g_slabCount = 0;
  g_current[0] = g_current[1] = -1;
  g_used = 0;
  g_pendingHead = -1;
  g_init = 0;
  init_once();
}
