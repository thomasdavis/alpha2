/*
 * vaspace.c — where things live in the GPU's address space.
 *
 * WHAT: hands out GPU virtual addresses, and reserves the ranges that physical
 * memory gets mapped into.
 *
 * WHY separate from mapping.c: choosing an address and installing a mapping at
 * it are different decisions, and only one of them is ours. RM will not pick an
 * address for us -- mapping without DMA_OFFSET_FIXED fails -- so the allocator
 * here is load-bearing rather than a convenience, and it shares an address
 * space with RM's own context buffers.
 */
#include "memory.h"
#include "../aether/ioctl.h"

#include <string.h>

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

NvU64 gaia_va_take(NvU64 size) {
  if (!g_vaNext) g_vaNext = GAIA_VA_BASE;
  /* 64 KiB granularity keeps every mapping comfortably page-aligned for any
   * page size RM might choose. */
  const NvU64 align = 64 * 1024;
  NvU64 at = (g_vaNext + align - 1) & ~(align - 1);
  g_vaNext = at + ((size + align - 1) & ~(align - 1));
  return at;
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

