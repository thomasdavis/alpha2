/*
 * memory.c — see memory.h.
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

/* NVOS46 — map into the GPU address space. sdk/nvidia/inc/nvos.h. */
typedef struct {
  NvHandle hClient;
  NvHandle hDevice;
  NvHandle hDma;
  NvHandle hMemory;
  NvU64 offset __attribute__((aligned(8)));
  NvU64 length __attribute__((aligned(8)));
  NvV32 flags;
  NvV32 flags2;
  NvV32 kindOverride;
  NvU64 dmaOffset __attribute__((aligned(8)));
  NvV32 status;
} NVOS46_PARAMETERS;

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

/* NVOS46_FLAGS_ACCESS_READ_WRITE is 0, so no flags are needed for the common
 * mapping. Named anyway so the zero is intentional rather than accidental. */
#define MAP_FLAGS_ACCESS_READ_WRITE 0x0
/* NVOS46_FLAGS_DMA_OFFSET_FIXED is bit 15 (field 15:15 in nvos.h). Worth stating
 * because bit 16 is DISABLE_ENCRYPTION and an off-by-one silently asks for
 * something entirely different. */
#define MAP_FLAGS_DMA_OFFSET_FIXED (1u << 15)

/* GPU virtual addresses are handed out from this base upward.
 *
 * Determined by probe, and every part of it mattered:
 *   base 0          -> NV_ERR_INVALID_ARGUMENT   (the null page)
 *   base 0x100000000-> NV_ERR_NO_MEMORY          (outside the default VA space)
 *   base 0x200000   -> NV_OK
 * and at any base, mapping WITHOUT DMA_OFFSET_FIXED also fails with
 * NV_ERR_INVALID_ARGUMENT -- RM will not pick an address for us here, it only
 * honours one we name. */
/* A working Vulkan driver places its GPFIFO around 0x04020000. We were handing
 * out addresses from 0x200000 -- which MAPS fine, but mapping succeeding is not
 * the same as the fetch engine being able to read from there. Moved to match
 * the reference range. */
/*
 * AND WE ARE NOT THE ONLY ONE ALLOCATING IN THIS ADDRESS SPACE.
 *
 * RM maps the GR context buffers into the channel's VA space when the compute
 * object is created, choosing its own addresses -- and a low base is exactly
 * where an allocator naturally starts. Handing out 0x04000000 upward through
 * DMA_OFFSET_FIXED, which forces a mapping at an address we name rather than
 * asking for a free one, can therefore land on top of RM's context mappings.
 * That would look precisely like what we see: everything works until a CTA
 * launches, because nothing before that reads the GR context.
 *
 * So the base moves clear of anything an allocator would pick first, but not so
 * far that it leaves the usable range. 16 TiB was tried and is OUTSIDE it: the
 * mappings succeed and then the GPU cannot even fetch the pushbuffer, reporting
 * ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT (31) with GP_GET stuck at zero. Useful
 * calibration in its own right -- it is the first error code other than 13 this
 * path has produced, which proves the notifier distinguishes causes rather than
 * reporting one catch-all.
 *
 * 32 GiB is inside the range (an early probe mapped there cleanly) and still far
 * above where RM hands itself context buffers.
 */
#define GAIA_VA_BASE 0x0000000800000000ULL

/*
 * GPU virtual addresses are handed out by a bump allocator.
 *
 * The first version mapped every buffer at GAIA_VA_BASE, which works for
 * exactly one buffer and then silently aliases: the second allocation lands on
 * the first. It surfaced as the channel failing to open, because a channel
 * needs three GPU-visible buffers at once -- a single-buffer test could never
 * have caught it.
 *
 * Bump-only, no reuse. Freed ranges are not recycled, which is fine while
 * allocation counts are small and honest about being a placeholder.
 */
static NvU64 g_vaNext;

static NvU64 gaia_va_take(NvU64 size) {
  if (!g_vaNext) g_vaNext = GAIA_VA_BASE;
  /* 64 KiB granularity keeps every mapping comfortably page-aligned for any
   * page size RM might choose. */
  const NvU64 align = 64 * 1024;
  NvU64 at = (g_vaNext + align - 1) & ~(align - 1);
  g_vaNext = at + ((size + align - 1) & ~(align - 1));
  return at;
}

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

/*
 * Reserve a GPU virtual address range.
 *
 * The VA space object (FERMI_VASPACE_A) is NOT what NVOS46 wants in hDma --
 * passing it, or the device, or the subdevice, all returned
 * NV_ERR_INVALID_OBJECT_HANDLE on an RTX 3070. What it wants is an
 * NV50_MEMORY_VIRTUAL object: a reserved *range* within a VA space. So GPU
 * mapping is two steps, not one -- reserve the range, then map physical pages
 * into it -- which is the same shape as Vulkan's separate VkDeviceMemory and
 * virtual-address concepts, just with the seam exposed.
 */
int gaia_reserve_va(aether_device *d, NvHandle *out, NvU64 base, NvU64 size) {
  /* NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS is declared in cl0070.h and therefore
   * belongs to class 0x70 (NV01_MEMORY_VIRTUAL). Pairing it with
   * NV50_MEMORY_VIRTUAL (0x50a0) returns NV_ERR_INVALID_ARGUMENT -- match
   * structs to the header that declares them.
   *
   * `limit` is INCLUSIVE. The parent must be the DEVICE: with parent=client or
   * parent=vaspace RM answers NV_ERR_INVALID_OBJECT_PARENT (0x36), which is how
   * we told a bad parent apart from bad params. */
  struct {
    NvU64 offset __attribute__((aligned(8)));
    NvU64 limit __attribute__((aligned(8)));
    NvHandle hVASpace;
  } p;
  memset(&p, 0, sizeof p);
  p.offset = base;
  p.limit = base + size - 1;
  p.hVASpace = d->vaspace;
  return aether_alloc(d, d->device, out, NV01_MEMORY_VIRTUAL, &p, sizeof p);
}

int gaia_map_gpu_at(aether_device *d, gaia_buffer *b, NvHandle hDma, NvU64 at) {
  NVOS46_PARAMETERS p;
  memset(&p, 0, sizeof p);
  p.hClient = d->client;
  p.hDevice = d->device;
  /* hDma names the address space. RM answered NV_ERR_INVALID_OBJECT_HANDLE for
   * the vaspace handle, so which object it wants here is decided empirically --
   * gaia_map_gpu_as() lets the probe try each candidate. */
  p.hDma = hDma;
  p.hMemory = b->handle;
  p.offset = 0;   /* offset within the MEMORY object */
  p.length = b->size;
  /* dmaOffset is IN as well as OUT here: with DMA_OFFSET_FIXED we tell RM the
   * address, and it echoes it back. */
  p.flags = MAP_FLAGS_ACCESS_READ_WRITE | MAP_FLAGS_DMA_OFFSET_FIXED;
  p.dmaOffset = at;

  int rc = aether_ioctl(d->ctlFd, NV_ESC_RM_MAP_MEMORY_DMA, &p, sizeof p);
  if (rc < 0) return rc;
  if (p.status != NV_OK) return (int)p.status;

  b->gpuAddr = p.dmaOffset;
  return 0;
}

int gaia_map_gpu(aether_device *d, gaia_buffer *b) {
  /* Two steps, because GPU mapping genuinely is two things: reserve an address
   * range, then place physical pages at a named address inside it. */
  const NvU64 at = gaia_va_take(b->size);
  NvHandle va = 0;
  int rc = gaia_reserve_va(d, &va, at, b->size);
  if (rc != 0) return rc;

  rc = gaia_map_gpu_at(d, b, va, at);
  if (rc != 0) {
    aether_free(d, va);
    return rc;
  }
  b->vaHandle = va;
  return 0;
}

int gaia_map_host(aether_device *d, gaia_buffer *b) {
  /*
   * Host mapping is TWO steps, and the first one is easy to mistake for the
   * whole thing.
   *
   * NV_ESC_RM_MAP_MEMORY takes NVOS33 wrapped with a file descriptor
   * (nv-unix-nvos-params-wrappers.h). It returns NV_OK and fills in
   * pLinearAddress -- but that value is NOT a pointer this process can
   * dereference. Observed on an RTX 3070: NV_OK with pLinearAddress
   * 0xb0190000, and the very first store to it segfaulted.
   *
   * What the call actually does is arrange a mapping that userspace then
   * completes with mmap() on the fd it was handed. The fd must be a FRESH
   * descriptor -- RM associates the mapping with it -- so we open one per
   * buffer rather than reusing d->gpuFd.
   *
   * The lesson generalises past this call: NV_OK from RM means "the request
   * was accepted", not "you now have what you asked for".
   */
  /*
   * THE MAPPING FD MUST MATCH THE APERTURE. This took an ioctl interposer on a
   * working CUDA process to find, and it is invisible in the headers:
   *
   *   video memory  -> /dev/nvidiaN   (the BAR1 aperture on that GPU)
   *   system memory -> /dev/nvidiactl (the control node)
   *
   * Mapping system memory through the device node returns
   * NV_ERR_INVALID_ARGUMENT for every combination of allocation attribute and
   * map flag -- which reads as "bad parameters" and is really "wrong file
   * descriptor". That single mistake made host-visible system memory look
   * impossible for four rounds of probing.
   */
  /*
   * AND THE FLAGS ARE NOT OPTIONAL EITHER.
   *
   * Left at zero, NVOS33_FLAGS_CACHING_TYPE (bits 25:23) is 0 = CACHED, so the
   * host mapping is ordinary write-back memory. Everything then LOOKS correct
   * from userspace -- a store followed by a load returns what was stored -- but
   * the value is sitting in a CPU cache line and the GPU never sees it. That is
   * the perfect crime for a doorbell protocol: GP_PUT reads back as the value we
   * wrote, the GPU's fetch engine reads the stale memory behind it, GP_GET never
   * advances, and no error is raised anywhere because nothing has gone wrong as
   * far as either side can tell.
   *
   * The values are what a working CUDA process passes, observed with the ioctl
   * interposer in tools/rm_spy.c and decoded against nvos.h:
   *
   *   sysmem  0x030c8000  MAPPING=DIRECT(1)     CACHING=DEFAULT(6)
   *   vidmem  0x010d0000  MAPPING=REFLECTED(2)  CACHING=WRITECOMBINED(2)
   *
   * (Both also set MAP_FIXED and RESERVE_ON_UNMAP, bits 18 and 19. Those belong
   * to CUDA's own VA-reservation scheme -- it supplies the address -- and we do
   * not, so we leave them clear and let mmap() choose.)
   *
   * WRITECOMBINED rather than UNCACHED for video memory is the interesting
   * choice: write-combining still buffers stores, so the sfence in hermes_ring
   * is doing real work rather than being decorative.
   */
  char path[32];
  if (b->location == GAIA_SYSMEM) {
    snprintf(path, sizeof path, "/dev/nvidiactl");
  } else {
    /* minor, not index: the two differ whenever the process cannot see every
     * card the driver knows about, which is the normal case in a container. */
    snprintf(path, sizeof path, "/dev/nvidia%d", d->minor);
  }
  int fd = open(path, O_RDWR | O_CLOEXEC);
  if (fd < 0) return -1;
  /* Every secondary descriptor is registered against the control fd, which is
   * what a working driver does for each fd it opens. */
  aether_register_fd(d, fd);

  struct {
    NVOS33_PARAMETERS params;
    int fd;
  } w;
  memset(&w, 0, sizeof w);
  w.params.hClient = d->client;
  w.params.hDevice = d->device;
  w.params.hMemory = b->handle;
  w.params.offset = 0;
  w.params.length = b->size;
  w.params.flags = b->mapFlags ? b->mapFlags
                   : b->location == GAIA_SYSMEM ? GAIA_MAP_FLAGS_SYSMEM
                                                : GAIA_MAP_FLAGS_VIDMEM;
  w.fd = fd;

  int rc = aether_ioctl(d->ctlFd, NV_ESC_RM_MAP_MEMORY, &w, sizeof w);
  if (rc < 0) { close(fd); return rc; }
  if (w.params.status != NV_OK) { close(fd); return (int)w.params.status; }

  void *p = mmap(NULL, (size_t)b->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) { close(fd); return -2; }

  b->hostPtr = p;
  b->hostFd = fd;
  return 0;
}

void gaia_free(aether_device *d, gaia_buffer *b) {
  /* Order matters: mappings reference the object, so they go first. */
  if (b->hostPtr) {
    munmap(b->hostPtr, (size_t)b->size);
    b->hostPtr = NULL;
  }
  if (b->hostFd >= 0) {
    close(b->hostFd);
    b->hostFd = -1;
  }
  if (b->vaHandle) {
    aether_free(d, b->vaHandle);
    b->vaHandle = 0;
  }
  if (b->gpuAddr) {
    NVOS46_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hClient = d->client;
    p.hDevice = d->device;
    p.hDma = d->vaspace;
    p.hMemory = b->handle;
    p.dmaOffset = b->gpuAddr;
    aether_ioctl(d->ctlFd, NV_ESC_RM_UNMAP_MEMORY_DMA, &p, sizeof p);
    b->gpuAddr = 0;
  }
  if (b->handle) {
    aether_free(d, b->handle);
    b->handle = 0;
  }
  b->size = 0;
}
