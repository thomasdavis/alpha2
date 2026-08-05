/*
 * alloc.c — obtaining memory, and the address space it will live in.
 *
 * WHAT: allocate video or system memory, and hand out the GPU virtual
 * addresses buffers get mapped at. Mapping itself is in mapping.c.
 */
#include "memory.h"
#include "../aether/ioctl.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* NV_MEMORY_ALLOCATION_PARAMS — sdk/nvidia/inc/nvos.h. Layout asserted in
 * gaia_test.c. */
typedef struct {
  NvU32 owner;
  NvU32 type;
  NvU32 flags;
  NvU32 width;
  NvU32 height;
  int32_t pitch;
  NvU32 attr;
  NvU32 attr2;
  NvU32 format;
  NvU32 comprCovg;
  NvU32 zcullCovg;
  NvU64 rangeLo __attribute__((aligned(8)));
  NvU64 rangeHi __attribute__((aligned(8)));
  NvU64 size __attribute__((aligned(8)));
  NvU64 alignment __attribute__((aligned(8)));
  NvU64 offset __attribute__((aligned(8)));
  NvU64 limit __attribute__((aligned(8)));
  NvP64 address __attribute__((aligned(8)));
  NvU32 ctagOffset;
  NvHandle hVASpace;
  NvU32 internalflags;
  NvU32 tag;
  int32_t numaNode;
} NV_MEMORY_ALLOCATION_PARAMS;

/*
 * The attr word is bit-packed. Positions from nvos.h:
 *   PAGE_SIZE    24:23
 *   LOCATION     26:25
 *   PHYSICALITY  28:27
 *   COHERENCY    31:29
 * The values themselves are the small constants NVOS32_ATTR_*_*, which have to
 * be shifted into place — a mistake here yields NV_ERR_INVALID_ARGUMENT with no
 * indication of which field was wrong, so the shifts are named rather than
 * inlined.
 */
#define ATTR_LOCATION_SHIFT 25
#define ATTR_PHYSICALITY_SHIFT 27
#define ATTR_COHERENCY_SHIFT 29

#define ATTR_LOCATION_VIDMEM 0
#define ATTR_LOCATION_PCI 1
#define ATTR_PHYSICALITY_CONTIGUOUS 2
/*
 * NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS.
 *
 * Contiguity is a real constraint and a small one: the kernel will not hand out
 * more than MAX_ORDER of physically adjacent pages, which is 4 MiB here, and
 * that is where SLAB_BYTES comes from. It is fine for a slab and fatal for a
 * TENSOR that is bigger than one -- a 105M-parameter model's embedding table is
 * 12,288 x 640 floats, 31.5 MB, and it simply cannot be allocated contiguously:
 * "helios: allocation of 7864320 floats failed", with nothing wrong but the
 * request.
 *
 * Nothing needs the contiguity. The GPU reaches this memory through page tables
 * that NVOS46 fills in, and scattered pages are what those are for; only a
 * caller doing physical addressing would care, and there is none.
 */
#define ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS 3
#define ATTR_COHERENCY_CACHED 1
#define ATTR_COHERENCY_WRITE_COMBINE 2

/* NVOS32_ALLOC_FLAGS_*, nvos.h */
#define ALLOC_FLAGS_MAP_NOT_REQUIRED 0x00008000

/* Allocate with a caller-chosen attr word. Exposed so the coherency of buffers
 * the GPU POLLS (the GPFIFO ring, the pushbuffer) can be chosen deliberately
 * rather than inheriting the default. */
int gaia_alloc_attr(aether_device *d, gaia_buffer *b, NvU64 size, NvU32 attr) {
  memset(b, 0, sizeof *b);
  b->hostFd = -1;
  b->location = GAIA_VIDMEM;
  NV_MEMORY_ALLOCATION_PARAMS p;
  memset(&p, 0, sizeof p);
  p.owner = 0x48454c49;
  p.size = size;
  p.alignment = 4096;
  p.flags = ALLOC_FLAGS_MAP_NOT_REQUIRED;
  p.attr = attr;
  int rc = aether_alloc(d, d->device, &b->handle, NV01_MEMORY_LOCAL_USER, &p, sizeof p);
  if (rc != 0) return rc;
  b->size = size;
  return 0;
}

/*
 * WRITE-COMBINED IS A ONE-WAY STREET, and which way matters enormously.
 *
 * Write-combining buffers CPU stores and flushes them in bursts, which is why
 * it is the right choice for anything the host writes and the GPU reads -- the
 * pushbuffer, the QMD, the constant bank. It says nothing kind about READS. A
 * CPU read of write-combined memory bypasses the cache entirely: one bus
 * transaction per access, no prefetch, no line reuse.
 *
 * Measured on this hardware over 2048 floats (micro-host-memory.mjs):
 *
 *     read  write-combined    225.7 us      plain memory   1.4 us    161x
 *     copy  wc -> wc          387.9 us      plain          1.0 us    388x
 *     write write-combined      1.2 us      plain          1.0 us    1.2x
 *
 * Writes are free; reads are not. TENSORS are read by the host constantly --
 * every broadcast, slice, concatenation and permutation walks them, and so does
 * every CPU fallback in autograd -- so a tensor wants the opposite trade from a
 * pushbuffer.
 *
 * Cached is safe here for the same reason CUDA's pinned host memory is
 * cacheable by default: on x86 the PCIe root complex snoops, so DMA in either
 * direction is coherent with the CPU caches without explicit maintenance. That
 * is a property of the platform, not of this code, which is why the known-answer
 * suite -- host writes an input, kernel reads it, host reads the result -- is
 * what actually proves it rather than this comment.
 */
int gaia_alloc_cached(aether_device *d, gaia_buffer *b, NvU64 size,
                      gaia_location where, int cached) {
  memset(b, 0, sizeof *b);
  b->hostFd = -1;

  NV_MEMORY_ALLOCATION_PARAMS p;
  memset(&p, 0, sizeof p);
  p.owner = 0x48454c49; /* 'HELI' — shows up in RM traces as ours */
  p.size = size;
  p.alignment = 4096;
  p.flags = ALLOC_FLAGS_MAP_NOT_REQUIRED;

  const NvU32 location =
      (where == GAIA_VIDMEM) ? ATTR_LOCATION_VIDMEM : ATTR_LOCATION_PCI;
  /* Contiguous keeps the first bridge simple: one physical range, one GPU
   * mapping, no scatter-gather to get wrong. */
  /* Contiguous while it is cheap, scattered when it must be. Small allocations
   * keep the proven path exactly; only requests past the kernel's contiguity
   * ceiling change, and for those the alternative is failing. */
  const NvU32 physicality = size > (4ull << 20) ? ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS
                                                : ATTR_PHYSICALITY_CONTIGUOUS;
  p.attr = (location << ATTR_LOCATION_SHIFT) |
           (physicality << ATTR_PHYSICALITY_SHIFT) |
           ((cached ? ATTR_COHERENCY_CACHED : ATTR_COHERENCY_WRITE_COMBINE)
            << ATTR_COHERENCY_SHIFT);

  const NvV32 cls =
      (where == GAIA_VIDMEM) ? NV01_MEMORY_LOCAL_USER : NV01_MEMORY_SYSTEM;

  int rc = aether_alloc(d, d->device, &b->handle, cls, &p, sizeof p);
  if (rc != 0) return rc;

  b->size = size;
  b->location = where;   /* remembered: it decides the mapping fd */
  b->cached = cached;    /* remembered: it decides the mapping's caching bits */
  return 0;
}

/* Write-combined, which is what every caller wanted before tensors needed to be
 * read. Unchanged behaviour for all of them. */
int gaia_alloc(aether_device *d, gaia_buffer *b, NvU64 size, gaia_location where) {
  return gaia_alloc_cached(d, b, size, where, 0);
}

