/*
 * spy_memory.c — knowing which addresses belong to the GPU, and reading them.
 *
 * WHAT: the list of mapped regions the driver handed this process, and a
 * self-read through /proc/self/mem.
 *
 * WHY /proc/self/mem RATHER THAN A PLAIN DEREFERENCE: the scanner walks
 * addresses it inferred rather than addresses it was given, so a wrong guess is
 * expected and must not be fatal. Reading through the file descriptor turns a
 * bad address into a short read instead of a segmentation fault, which is the
 * difference between a tool that reports what it found and one that dies on the
 * first thing it did not understand.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not write. This is a spy on a
 * running driver, and a tool that can perturb what it observes is a tool whose
 * observations cannot be trusted.
 */
#include "spy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

struct spy_region spy_regions[MAX_REGIONS];
int spy_nregions;
FILE *L;

int spy_in_gpu_region(uint64_t a, uint64_t len) {
  for (int i = 0; i < spy_nregions; i++)
    if (a >= spy_regions[i].lo && a + len <= spy_regions[i].hi) return 1;
  return 0;
}

void spy_load_regions(void) {
  spy_nregions = 0;
  FILE *m = fopen("/proc/self/maps", "r");
  if (!m) return;
  char line[512];
  while (spy_nregions < MAX_REGIONS && fgets(line, sizeof line, m)) {
    uint64_t lo, hi;
    char perms[8];
    if (sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) != 3) continue;
    if (perms[0] != 'r' || perms[1] != 'w') continue;
    /*
     * GPU-visible regions only, and this restriction is not optional.
     *
     * Dropping it while hunting for the (wrong) PCAS signature seemed harmless;
     * re-running with the correct inline-QMD signature over ALL memory then
     * produced hits whose "QMD" was uniformly high-entropy -- random heap that
     * happens to match 13 bits of address and a plausible count. A pushbuffer
     * is by definition memory the GPU reads, so it is in the band. Signature
     * AND provenance; neither alone is enough.
     */
    if (lo < GPU_VA_LO || lo >= GPU_VA_HI) continue;
    if (hi - lo > (256u << 20)) continue;
    spy_regions[spy_nregions].lo = lo;
    spy_regions[spy_nregions].hi = hi;
    spy_nregions++;
  }
  fclose(m);
}

/*
 * Read our own memory through /proc/self/mem rather than by dereferencing.
 *
 * Scanning every writable region directly killed the traced process: some of
 * those mappings fault on read (guard pages, device mappings with restricted
 * access), and a SIGSEGV inside the scanner thread takes the whole program with
 * it. pread on /proc/self/mem returns -1 for exactly those pages instead, which
 * turns an unreadable region into a skipped region.
 */
static int memfd = -1;

long spy_read_self(uint64_t addr, void *dst, size_t len) {
  if (memfd < 0) memfd = open("/proc/self/mem", O_RDONLY);
  if (memfd < 0) return -1;
  return pread(memfd, dst, len, (off_t)addr);
}

