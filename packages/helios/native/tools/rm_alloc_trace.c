/*
 * rm_alloc_trace.c — an LD_PRELOAD shim that prints every RM memory allocation.
 *
 * WHY: HELIOS_VIDMEM fails with NV_ERR_INSUFFICIENT_RESOURCES (81) on every
 * video-memory slab after the first, at every size from 4 MiB down to 128 KiB,
 * with 7.8 GiB free on the card. So it is not capacity and not the size -- it is
 * something in NV_MEMORY_ALLOCATION_PARAMS that RM does not like for
 * NV01_MEMORY_LOCAL_USER, and guessing the field costs a build and a run each
 * time. One guess (PHYSICALITY) has already been spent and was wrong.
 *
 * This stack has answered exactly this kind of question before by observing a
 * working process rather than reasoning about the header -- gaia/memory.h
 * records "observed flags for the 64 KiB usermode doorbell page in a working
 * CUDA process". Same method: the pod image carries PyTorch and CUDA, so run a
 * one-line torch allocation under this shim, run our own backend under it too,
 * and diff the parameter words.
 *
 * It only reads. It forwards every call untouched and prints to stderr.
 *
 *   cc -shared -fPIC -o rm_alloc_trace.so rm_alloc_trace.c -ldl
 *   LD_PRELOAD=./rm_alloc_trace.so python -c "import torch; torch.zeros(1<<20, device='cuda')"
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>

/* NV_ESC_RM_ALLOC, and the two memory classes we care about. */
#define ESC_RM_ALLOC 0x2B
#define CLASS_LOCAL_USER 0x40
#define CLASS_SYSTEM 0x3e

typedef struct {
  uint32_t hRoot, hObjectParent, hObjectNew;
  int32_t hClass;
  uint64_t pAllocParms __attribute__((aligned(8)));
  uint32_t paramsSize;
  int32_t status;
} NVOS21;

typedef struct {
  uint32_t owner, type, flags, width, height;
  int32_t pitch;
  uint32_t attr, attr2, format, comprCovg, zcullCovg;
  uint64_t rangeLo __attribute__((aligned(8)));
  uint64_t rangeHi __attribute__((aligned(8)));
  uint64_t size __attribute__((aligned(8)));
  uint64_t alignment __attribute__((aligned(8)));
  uint64_t offset __attribute__((aligned(8)));
  uint64_t limit __attribute__((aligned(8)));
  uint64_t address __attribute__((aligned(8)));
  uint32_t ctagOffset, hVASpace, internalflags, tag;
  int32_t numaNode;
} MEMPARAMS;

static int (*real_ioctl)(int, unsigned long, ...);

int ioctl(int fd, unsigned long req, ...) {
  va_list ap;
  va_start(ap, req);
  void *arg = va_arg(ap, void *);
  va_end(ap);
  if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");

  const int isAlloc = (req & 0xff) == ESC_RM_ALLOC;
  const int rc = real_ioctl(fd, req, arg);

  if (isAlloc && arg) {
    NVOS21 *p = (NVOS21 *)arg;
    if (p->hClass == CLASS_LOCAL_USER || p->hClass == CLASS_SYSTEM) {
      MEMPARAMS *m = (MEMPARAMS *)(uintptr_t)p->pAllocParms;
      /* The attr word decoded the way gaia/alloc.c packs it, so the two sides
       * can be compared field by field rather than as one hex number. */
      const unsigned loc = m ? (m->attr >> 25) & 3u : 0;
      const unsigned phys = m ? (m->attr >> 27) & 3u : 0;
      const unsigned coh = m ? (m->attr >> 29) & 7u : 0;
      const unsigned pgsz = m ? (m->attr >> 23) & 3u : 0;
      fprintf(stderr,
              "[rm] class=0x%02x status=%d size=%llu align=%llu flags=0x%08x "
              "attr=0x%08x{page=%u loc=%u phys=%u coh=%u} attr2=0x%08x "
              "type=%u fmt=%u hVASpace=0x%08x range=[%llu,%llu] parent=0x%08x\n",
              p->hClass, p->status,
              m ? (unsigned long long)m->size : 0ull,
              m ? (unsigned long long)m->alignment : 0ull,
              m ? m->flags : 0u, m ? m->attr : 0u, pgsz, loc, phys, coh,
              m ? m->attr2 : 0u, m ? m->type : 0u, m ? m->format : 0u,
              m ? m->hVASpace : 0u,
              m ? (unsigned long long)m->rangeLo : 0ull,
              m ? (unsigned long long)m->rangeHi : 0ull,
              p->hObjectParent);
    }
  }
  return rc;
}
