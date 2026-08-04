/*
 * submit_probe.c — does the host engine fetch ANY GPFIFO entry?
 *
 * WHAT: brings up a channel and submits three things in turn, watching GP_GET
 * after each: an empty entry, a NOP method, and a semaphore release.
 *
 * WHY these three: they separate three failures that all look identical from
 * outside. If GP_GET advances on the empty entry, the host engine is fetching
 * and the problem is in what the methods say. If it advances on the NOP but not
 * the semaphore, the semaphore encoding is wrong. If it never moves at all, the
 * problem is upstream of the pushbuffer entirely -- the channel is not running,
 * and nothing about method encoding matters yet.
 *
 * A sweep over parameter values could not have told these apart, which is the
 * whole point: a probe that fails identically everywhere is evidence you are
 * varying the wrong thing.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: it does not try to fix anything. It
 * prints the addresses and the counters and stops.
 */
#include "../hermes/pushbuffer.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static NvU64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (NvU64)ts.tv_sec * 1000000000ull + (NvU64)ts.tv_nsec;
}

/* Ring, then watch GP_GET for a bounded window. Returns the final GP_GET. */
static NvU32 ring_and_watch(hermes_channel *c, const char *what) {
  volatile NvU32 *page = (volatile NvU32 *)c->userd.hostPtr;
  volatile NvU32 *userd = page + c->userdSlot / 4;
  hermes_ring(c, page, c->doorbell, c->token);

  const NvU64 deadline = now_ns() + 250000000ull; /* 250 ms */
  NvU32 get = 0;
  while (now_ns() < deadline) {
    get = userd[HERMES_USERD_GP_GET / 4];
    if (get == c->put) break;
  }
  printf("  %-24s GP_PUT=%u GP_GET=%u %s\n", what, c->put, get,
         get == c->put ? "<-- FETCHED" : "(not fetched)");
  return get;
}

int main(void) {
  aether_device d;
  if (aether_device_open(&d, 0) != 0) {
    printf("device open failed at %s (%d)\n",
           d.failStage ? d.failStage : "open /dev/nvidiactl", d.failStatus);
    return 1;
  }
  printf("device: minor=%d gpuId=0x%08x\n", d.minor, d.gpuId);

  hermes_channel c;
  if (hermes_channel_open(&d, &c) != 0) {
    printf("channel open failed at %s (0x%x)\n", c.failStage, c.failStatus);
    return 1;
  }

  printf("channel: token=0x%x entries=%u\n", c.token, c.gpfifoEntries);
  printf("  gpfifo     gpu=0x%016llx host=%p\n",
         (unsigned long long)c.gpfifo.gpuAddr, c.gpfifo.hostPtr);
  printf("  pushbuffer gpu=0x%016llx host=%p\n",
         (unsigned long long)c.pushbuffer.gpuAddr, c.pushbuffer.hostPtr);
  printf("  userd      gpu=0x%016llx host=%p\n",
         (unsigned long long)c.userd.gpuAddr, c.userd.hostPtr);
  printf("  usermode   host=%p  TIME_0=0x%08x\n", c.usermode.hostPtr,
         c.doorbell[0x80 / 4]);

  /* 1. An empty entry: a GPFIFO slot with LENGTH 0 and no pushbuffer behind it.
   *    The host engine still has to fetch and retire it. */
  {
    NvU32 *ring = (NvU32 *)c.gpfifo.hostPtr;
    ring[c.put * 2 + 0] = 0;
    ring[c.put * 2 + 1] = 0;
    c.put++;
    __asm__ __volatile__("sfence" ::: "memory");
    ring_and_watch(&c, "empty entry");
  }

  /* 2. A NOP method. Opcode 0 with count 0 is a no-op the host consumes. */
  {
    hermes_begin(&c);
    hermes_data(&c, 0x00000000u);
    hermes_submit(&d, &c);
    ring_and_watch(&c, "NOP method");
  }

  /* 3. The semaphore release we actually care about. */
  {
    gaia_buffer sem;
    if (gaia_alloc(&d, &sem, 4096, GAIA_SYSMEM) == 0 &&
        gaia_map_gpu(&d, &sem) == 0 && gaia_map_host(&d, &sem) == 0) {
      ((volatile NvU32 *)sem.hostPtr)[0] = 0;
      printf("  semaphore  gpu=0x%016llx\n", (unsigned long long)sem.gpuAddr);

      hermes_begin(&c);
      hermes_semaphore_release(&c, sem.gpuAddr, 0xcafebabeu);
      hermes_submit(&d, &c);
      ring_and_watch(&c, "semaphore release");
      printf("  semaphore value = 0x%08x\n", ((volatile NvU32 *)sem.hostPtr)[0]);
      gaia_free(&d, &sem);
    }
  }

  {
    const NvU32 *n = (const NvU32 *)c.errnotif.hostPtr;
    printf("  errnotif: %08x %08x %08x %08x\n", n[0], n[1], n[2], n[3]);
  }

  /*
   * Dump the WHOLE page, not just the 512 bytes RM let us request.
   *
   * RM refuses a map longer than 512 bytes (NV_ERR_INVALID_LIMIT), but mmap
   * still hands back a full page, and the open question is whether offset 0 of
   * that page is OUR channel's 512-byte slot or the base of a page we share
   * with seven other channels. Printing only non-zero rows lets the data answer
   * it: whichever offset our GP_PUT of 3 shows up at is where we are really
   * writing.
   */
  printf("  USERD page (slot should be 0x%x), non-zero rows only:\n", c.userdSlot);
  const NvU32 *u = (const NvU32 *)c.userd.hostPtr;
  for (int i = 0; i < 1024; i += 8) {
    int nz = 0;
    for (int k = 0; k < 8; k++) if (u[i + k]) nz = 1;
    if (!nz) continue;
    printf("    +0x%03x ", i * 4);
    for (int k = 0; k < 8; k++) printf("%08x ", u[i + k]);
    printf("\n");
  }

  hermes_channel_close(&d, &c);
  aether_device_close(&d);
  return 0;
}
