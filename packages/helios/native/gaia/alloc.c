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

int gaia_alloc(aether_device *d, gaia_buffer *b, NvU64 size, gaia_location where) {
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
  p.attr = (location << ATTR_LOCATION_SHIFT) |
           (ATTR_PHYSICALITY_CONTIGUOUS << ATTR_PHYSICALITY_SHIFT) |
           (ATTR_COHERENCY_WRITE_COMBINE << ATTR_COHERENCY_SHIFT);

  const NvV32 cls =
      (where == GAIA_VIDMEM) ? NV01_MEMORY_LOCAL_USER : NV01_MEMORY_SYSTEM;

  int rc = aether_alloc(d, d->device, &b->handle, cls, &p, sizeof p);
  if (rc != 0) return rc;

  b->size = size;
  b->location = where;   /* remembered: it decides the mapping fd */
  return 0;
}

