/*
 * slab_probe.c — how large a slab this machine will actually give us.
 *
 * WHAT: allocates, GPU-maps and host-maps system memory at descending sizes and
 * prints which sizes succeed.
 *
 * WHY it exists: the tensor pool carves many tensors out of one slab so that a
 * first-time allocation is a pointer bump rather than three ioctls (measured at
 * 802.3 us against 1.0 us from the free list). The slab size is therefore a
 * constant that decides how often the driver is touched at all -- and 64 MiB,
 * chosen by arithmetic, FAILED outright. gaia_alloc asks for PHYSICALLY
 * CONTIGUOUS pages, and a kernel will not hand out 64 MiB of those.
 *
 * A constant that a machine silently refuses is worse than a smaller one, so
 * this measures the ceiling instead of assuming it, and the number it prints is
 * the provenance for SLAB_BYTES in tensor.c.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no fallback and no retry logic of its
 * own. It reports what each size did. Choosing a policy from that is tensor.c's
 * job, and mixing the two would make the measurement depend on the policy it is
 * supposed to inform.
 */
#include "../gaia/memory.h"
#include "../aether/ioctl.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  aether_device dev;
  if (aether_device_open(&dev, 0) != 0) {
    printf("no device (%s)\n", dev.failStage ? dev.failStage : "?");
    return 1;
  }

  printf("%-10s %-8s %-8s %-9s\n", "size", "alloc", "map_gpu", "map_host");
  for (NvU64 size = 256ull << 20; size >= (4ull << 10); size >>= 1) {
    gaia_buffer b;
    const int a = gaia_alloc(&dev, &b, size, GAIA_SYSMEM);
    const int g = a == 0 ? gaia_map_gpu(&dev, &b) : -1;
    const int h = g == 0 ? gaia_map_host(&dev, &b) : -1;

    char label[16];
    if (size >= (1ull << 20)) snprintf(label, sizeof label, "%lluM", (unsigned long long)(size >> 20));
    else snprintf(label, sizeof label, "%lluK", (unsigned long long)(size >> 10));
    printf("%-10s %-8s %-8s %-9s\n", label,
           a == 0 ? "ok" : "FAIL", a != 0 ? "-" : (g == 0 ? "ok" : "FAIL"),
           g != 0 ? "-" : (h == 0 ? "ok" : "FAIL"));

    /* A touch through both mappings, because an allocation that maps and then
     * faults on use would otherwise read as success. */
    if (h == 0) {
      *(volatile NvU32 *)b.hostPtr = 0x5A5A5A5Au;
      if (*(volatile NvU32 *)b.hostPtr != 0x5A5A5A5Au) printf("  (host mapping does not hold a write)\n");
    }
    if (a == 0) gaia_free(&dev, &b);
  }

  aether_device_close(&dev);
  return 0;
}
