/*
 * mapping.c — turning an allocation into an address something can dereference.
 *
 * WHAT: reserve GPU virtual address ranges, map buffers into them, and map
 * buffers into this process.
 *
 * WHY separate from alloc.c: allocating gives you a memory OBJECT with no
 * address at all, and the two mappings that follow are independent of each
 * other and of the allocation. Conflating them is the first thing that goes
 * wrong when writing a driver, and keeping them in different files makes the
 * three-step shape hard to forget.
 *
 * The hard-won details -- which file descriptor each aperture maps through,
 * that NVOS33's flags are not optional, that NV_OK does not mean you have a
 * usable pointer -- are documented at the calls that carry them.
 */
#include "memory.h"
#include "../aether/ioctl.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

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

/* NVOS46_FLAGS_ACCESS_READ_WRITE is 0, so no flags are needed for the common
 * mapping. Named anyway so the zero is intentional rather than accidental. */
#define MAP_FLAGS_ACCESS_READ_WRITE 0x0
/* NVOS46_FLAGS_DMA_OFFSET_FIXED is bit 15 (field 15:15 in nvos.h). Worth stating
 * because bit 16 is DISABLE_ENCRYPTION and an off-by-one silently asks for
 * something entirely different. */
#define MAP_FLAGS_DMA_OFFSET_FIXED (1u << 15)

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
