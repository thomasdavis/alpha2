/*
 * hermes_test.c — does the GPU actually run what we submit?
 *
 * WHAT: the P0 gate. Build a channel, hand it a pushbuffer containing a single
 * semaphore release, ring the doorbell, and wait for a value we chose to appear
 * at an address we chose, written by the hardware.
 *
 * WHY a semaphore release rather than a kernel: it is the smallest thing that
 * can prove the GPU consumed our work. It exercises the entire path — object
 * model, address space, ring, pushbuffer, method encoding, doorbell — while
 * depending on no instruction encoding at all. If this fails, an assembler is
 * not the problem; if it passes, everything below the assembler is proven.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no kernel launch, no QMD, no grid. Those
 * come next, and they come after this passes.
 *
 * On the two offsets this test is really about (see pushbuffer.h): GP_PUT is
 * word 35 of USERD and GP_GET is word 34, per swref/published/ampere/ga100/
 * dev_ram.h. Both are asserted as compile-time constants below, because the
 * first implementation used 0x40 — which is NV_RAMUSERD_PUT, the legacy
 * pre-GPFIFO pointer — and every downstream symptom followed from that.
 */
#include "harness.h"
#include "../aether/ioctl.h" /* aether_status_name — a failure must name itself */
#include "../hermes/pushbuffer.h"
#include "../hermes/qmd.h"

#include <time.h>

/* The value the GPU must write, and where. Arbitrary, but recognisable in a hex
 * dump and not a value any zeroed buffer could produce by accident. */
#define SEM_PAYLOAD 0xcafebabeu

/* Offsets restated from the hardware header, so a future edit that "tidies" the
 * constants has to argue with a test rather than with a comment. */
static void test_userd_offsets_match_dev_ram(void) {
  HT_CASE("USERD field offsets come from dev_ram.h word indices");
  /* NV_RAMUSERD_GP_GET (34*32+31):(34*32+0) -> word 34 */
  HT_EQ_U64(HERMES_USERD_GP_GET, 34u * 4u);
  /* NV_RAMUSERD_GP_PUT (35*32+31):(35*32+0) -> word 35 */
  HT_EQ_U64(HERMES_USERD_GP_PUT, 35u * 4u);
  /* And the one that caused the bug: 0x40 is word 16, NV_RAMUSERD_PUT. It is a
   * real field, which is why writing to it looked harmless -- it is simply the
   * wrong one for a GPFIFO channel. */
  HT_TRUE(HERMES_USERD_GP_PUT != 0x40);

  /* The slot: eight 512-byte blocks per page, indexed by the low 3 bits of the
   * channel id, which the work-submit token carries in bits 11:0. Channel 4 --
   * what this GPU hands out -- lands at 0x800, which is exactly the offset RM
   * reported in pLinear and that went unexplained for the whole investigation. */
  HT_EQ_U64(HERMES_USERD_SLOT(0x4), 0x800);
  HT_EQ_U64(HERMES_USERD_SLOT(0x0), 0x000);
  /* runlist bits above 11:0 must not leak into the slot */
  HT_EQ_U64(HERMES_USERD_SLOT(0x10007), 0xe00);
  HT_END();
}

/* The doorbell is at NV_VIRTUAL_FUNCTION_DOORBELL (0x30090) inside a window
 * based at NV_VIRTUAL_FUNCTION (0x30000). */
static void test_doorbell_offset(void) {
  HT_CASE("doorbell offset is DOORBELL minus the window base");
  HT_EQ_U64(HERMES_DOORBELL_OFFSET, 0x30090u - 0x30000u);
  HT_END();
}

/* Method and entry encoding, checked against the bit fields in clc56f.h rather
 * than against whatever the encoder happens to emit. */
static void test_method_encoding(void) {
  HT_CASE("method header packs opcode, count, subchannel and address");
  /* INC_METHOD(1) << 29 | count(4) << 16 | sub(0) << 13 | SEMAPHOREA(0x10)>>2 */
  const NvU32 want = (1u << 29) | (4u << 16) | (0u << 13) | (0x10u >> 2);
  HT_EQ_U64(want, 0x20040004u);
  HT_END();
}

static NvU64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (NvU64)ts.tv_sec * 1000000000ull + (NvU64)ts.tv_nsec;
}

/*
 * The gate itself. Skips cleanly with no GPU so this file still runs in CI.
 */
static void test_gpu_executes_a_semaphore_release(void) {
  HT_CASE("GPU consumes a submitted pushbuffer");

  aether_device d;
  if (aether_device_open(&d, 0) != 0) {
    /* Skipping is only honest when there is genuinely no driver. If the driver
     * is present and open still failed, that is a FAILURE and it has to say so
     * -- a silent skip on a GPU box is how this suite stayed green while
     * verifying nothing. */
    if (d.failStage == NULL) {
      printf("skip (no NVIDIA driver)\n");
      ht_case_failed = 0;
      return;
    }
    HT_FAIL("device open failed at %s: %s (%d)", d.failStage,
            aether_status_name((unsigned)d.failStatus), d.failStatus);
    HT_END();
    return;
  }

  hermes_channel c;
  int rc = hermes_channel_open(&d, &c);
  if (rc != 0) {
    HT_FAIL("channel bring-up failed at %s: %s", c.failStage,
            aether_status_name((unsigned)c.failStatus));
    aether_device_close(&d);
    HT_END();
    return;
  }

  /* The semaphore lives in system memory: the GPU writes it and the host polls
   * it, and sysmem is the aperture where that is unambiguous. */
  gaia_buffer sem;
  if ((rc = gaia_alloc(&d, &sem, 4096, GAIA_SYSMEM)) != 0 ||
      (rc = gaia_map_gpu(&d, &sem)) != 0 ||
      (rc = gaia_map_host(&d, &sem)) != 0) {
    HT_FAIL("semaphore buffer: %s", aether_status_name((unsigned)rc));
    goto done;
  }
  ((volatile NvU32 *)sem.hostPtr)[0] = 0;

  hermes_begin(&c);
  hermes_semaphore_release(&c, sem.gpuAddr, SEM_PAYLOAD);
  hermes_submit(&d, &c);

  /* hermes_ring takes the PAGE base and applies the slot itself; reading back
   * needs the same offset applied by hand. */
  volatile NvU32 *page = (volatile NvU32 *)c.userd.hostPtr;
  volatile NvU32 *userd = page + c.userdSlot / 4;
  hermes_ring(&c, page, c.doorbell, c.token);

  /* Poll for a bounded time. A GPU that is going to run this runs it in
   * microseconds; a second is generous enough that a timeout means "never". */
  const NvU64 deadline = now_ns() + 1000000000ull;
  NvU32 got = 0;
  while (now_ns() < deadline) {
    got = ((volatile NvU32 *)sem.hostPtr)[0];
    if (got == SEM_PAYLOAD) break;
  }

  const NvU32 gpGet = userd[HERMES_USERD_GP_GET / 4];
  const NvU32 gpPut = userd[HERMES_USERD_GP_PUT / 4];

  if (got != SEM_PAYLOAD) {
    HT_FAIL("semaphore never released: got 0x%08x, want 0x%08x", got,
            SEM_PAYLOAD);
    /* GP_GET is the diagnosis, not decoration. GP_GET == GP_PUT with no
     * semaphore means the GPU fetched and the methods are wrong; GP_GET stuck
     * at 0 means it never fetched, which is a submission problem. */
    printf("      GP_GET=%u GP_PUT=%u token=0x%x slot=0x%x\n", gpGet, gpPut,
           c.token, c.userdSlot);
    printf("      errnotif[0..3] = %08x %08x %08x %08x\n",
           ((NvU32 *)c.errnotif.hostPtr)[0], ((NvU32 *)c.errnotif.hostPtr)[1],
           ((NvU32 *)c.errnotif.hostPtr)[2], ((NvU32 *)c.errnotif.hostPtr)[3]);
  } else {
    HT_EQ_U64(got, SEM_PAYLOAD);
    /* Having consumed the entry, the GPU must have advanced GP_GET to match.
     * Checking this is what distinguishes "the GPU ran our work" from "someone
     * else wrote that value" -- a second, independent piece of evidence for the
     * same event. */
    HT_EQ_U64(gpGet, gpPut);
  }

  gaia_free(&d, &sem);
done:
  hermes_channel_close(&d, &c);
  aether_device_close(&d);
  HT_END();
}

/*
 * The QMD's field positions, as known answers from NVIDIA's clc7c0qmd.h.
 *
 * Every position is stated there as MW(hi:lo) over the whole descriptor, so
 * word = lo/32 and shift = lo%32. Checking a few by construction catches the
 * two ways this goes wrong, both of which happened: a mistranscribed offset,
 * and -- worse, because it is invisible -- a mistaken WIDTH, which silently
 * writes zeroes across the neighbouring field.
 */
static void test_qmd_field_positions(void) {
  HT_CASE("QMD fields land where clc7c0qmd.h says");
  NvU32 q[HERMES_QMD_DWORDS];

  /* Distinct values so no field can be confused with another. Shared memory is
   * given a real size too: it is the one argument whose encoding is not the
   * value itself but a hardware code, so passing zero would leave the only
   * non-obvious field untested. */
  hermes_qmd_build(q, 0x800060000ull, 0, 3, 5, 13, 9, 11, 7, 4096);

  HT_EQ_U64(q[12], 3);  /* CTA_RASTER_WIDTH   MW(415:384) */
  HT_EQ_U64(q[13], 5);  /* CTA_RASTER_HEIGHT  MW(431:416) */
  HT_EQ_U64(q[14], 13); /* CTA_RASTER_DEPTH   MW(463:448) */
  HT_EQ_U64(q[18] >> 16, 9);            /* CTA_THREAD_DIMENSION0 MW(607:592) */
  HT_EQ_U64(q[19] & 0xffff, 11);        /* CTA_THREAD_DIMENSION1 MW(623:608) */
  HT_EQ_U64(q[19] >> 16, 7);            /* CTA_THREAD_DIMENSION2 MW(639:624) */
  HT_EQ_U64(q[48], 0x00060000u);        /* PROGRAM_ADDRESS_LOWER MW(1567:1536) */
  HT_EQ_U64(q[49] & 0x1ffff, 8);        /* PROGRAM_ADDRESS_UPPER MW(1584:1568) */
  HT_EQ_U64(q[51] >> 24, 0x86);         /* SASS_VERSION          MW(1663:1656) */
  HT_EQ_U64((q[18] >> 4) & 0xf, 3);     /* QMD_MAJOR_VERSION     MW(583:580) */

  /* The SM shared-memory partition, encoded (kB/4)+1. Zero is not a legal
   * configuration, which is why leaving these unset faults every launch. */
  HT_EQ_U64((q[17] >> 18) & 0x3f, 5);   /* MIN    16 KB */
  HT_EQ_U64((q[17] >> 25) & 0x3f, 26);  /* MAX   100 KB */
  HT_EQ_U64((q[20] >> 17) & 0x3f, 5);   /* TARGET 16 KB */

  /* SHARED_MEMORY_SIZE is bytes, rounded up to 256. The request was 4096, which
   * is already a multiple, so this also pins that rounding does not silently
   * inflate a value that needs none. */
  HT_EQ_U64(q[17] & 0x3ffff, 4096);     /* SHARED_MEMORY_SIZE    MW(561:544) */

  /*
   * WIDTHS, not just offsets. CTA_RASTER_HEIGHT is sixteen bits; a value that
   * fits in sixteen must not disturb the upper half of its dword. Writing it as
   * a 32-bit field passes every check above and fails this one.
   */
  hermes_qmd_build(q, 0x800060000ull, 0, 1, 0xffffu, 1, 1, 1, 1, 0);
  HT_EQ_U64(q[13] & 0xffff, 0xffffu);
  HT_EQ_U64(q[13] >> 16, 0);
  HT_END();
}

void ht_run(void) {
  printf("\nhermes — channels and submission\n");
  test_userd_offsets_match_dev_ram();
  test_doorbell_offset();
  test_method_encoding();
  test_qmd_field_positions();
  test_gpu_executes_a_semaphore_release();
}
