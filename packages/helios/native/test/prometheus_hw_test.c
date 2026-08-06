/*
 * prometheus_hw_test.c — running every registered kernel on the device.
 *
 * WHAT: allocate the buffers, write the code and parameters, launch, wait, and
 * judge. Split from prometheus_test.c, which now holds only the checks that
 * need no hardware.
 *
 * WHY THE SPLIT IS ALONG THAT LINE: the static checks run everywhere and must
 * never be skipped; this file SKIPS when there is no driver. Keeping them in
 * one file meant a machine without a GPU reported a suite that had quietly
 * stopped testing most of what it names.
 */
#include "harness.h"
#include "../aether/ioctl.h"
#include "../hermes/pushbuffer.h"
#include "../prometheus/kernel.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

/*
 * A sentinel the kernel's semaphore release writes, placed WELL past the output.
 *
 * It was at out+64 once, which is element 16 -- so a checker reporting "wrote 16
 * of 64 slots" was indistinguishable from a one-warp launch. Far enough away
 * that no kernel's output can reach it.
 *
 * RAISED to 128 KiB when the tensor-core GEMM arrived: a 64x64 tile of f32 is
 * 16 KiB and the cp.async f16 GEMM writes a whole tile, so the old 1 KiB fence
 * sat INSIDE the output. The output buffer grew to match; a fence a kernel can
 * reach is worse than no fence, because it turns a working kernel into a
 * hung-looking one. */
#define FENCE_OFFSET 0x20000u
#define FENCE_VALUE 0x5eed5eedu

static NvU64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (NvU64)ts.tv_sec * 1000000000ull + (NvU64)ts.tv_nsec;
}


/* Everything a launch needs, allocated once. */
typedef struct {
  gaia_buffer code, out, in, inB, inC, qmd, scratch, lmem;
} pr_buffers;

static int alloc_buffers(aether_device *d, pr_buffers *b) {
  int rc;
  memset(b, 0, sizeof *b);
  /* out holds the output BELOW the fence at 128 KiB, plus the fence — 256 KiB
   * covers a 64x256 f32 tile. in and inB hold the GEMM operands: a 64-row f16
   * tile at K up to a few thousand fits 64 KiB. The small kernels use a corner
   * of these and do not care that they grew. */
  if ((rc = gaia_alloc(d, &b->out, 0x40000, GAIA_SYSMEM)) != 0) return rc;
  if ((rc = gaia_map_gpu(d, &b->out)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->out)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->in, 0x10000, GAIA_SYSMEM)) != 0) return rc;
  if ((rc = gaia_map_gpu(d, &b->in)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->in)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->inC, 4096, GAIA_SYSMEM)) != 0) return rc;
  if ((rc = gaia_map_gpu(d, &b->inC)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->inC)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->inB, 0x10000, GAIA_SYSMEM)) != 0) return rc;
  if ((rc = gaia_map_gpu(d, &b->inB)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->inB)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->code, 4096, GAIA_VIDMEM)) != 0) return rc;
  if ((rc = gaia_map_gpu(d, &b->code)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->code)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->qmd, 4096, GAIA_VIDMEM)) != 0) return rc;
  if ((rc = gaia_map_gpu(d, &b->qmd)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->qmd)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->scratch, HERMES_QMD_SCRATCH_BYTES, GAIA_VIDMEM)) != 0)
    return rc;
  if ((rc = gaia_map_gpu(d, &b->scratch)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->scratch)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->lmem, 1024 * 1024, GAIA_VIDMEM)) != 0) return rc;
  return gaia_map_gpu(d, &b->lmem);
}

static void free_buffers(aether_device *d, pr_buffers *b) {
  gaia_free(d, &b->lmem);
  gaia_free(d, &b->scratch);
  gaia_free(d, &b->qmd);
  gaia_free(d, &b->code);
  gaia_free(d, &b->inB);
  gaia_free(d, &b->in);
  gaia_free(d, &b->out);
}

/* Engine init, once, in its own submission so it is consumed before any launch. */
static void init_engine(aether_device *d, hermes_channel *c, pr_buffers *b) {
  hermes_compute_config cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.classId = HERMES_COMPUTE_CLASS;
  cfg.spaVersion = HERMES_SPA_VERSION_SM86;
  cfg.sharedWindow = HERMES_SHARED_WINDOW_DEFAULT;
  cfg.localWindow = HERMES_LOCAL_WINDOW_DEFAULT;
  cfg.localMem = b->lmem.gpuAddr;
  cfg.localMemSize = b->lmem.size;
  cfg.smCount = HERMES_SM_COUNT_SM86;
  hermes_begin(c);
  hermes_compute_init(c, 1, &cfg);
  hermes_submit(d, c);
  hermes_ring(c, (volatile NvU32 *)c->userd.hostPtr, c->doorbell, c->token);
}

/* Fill the code buffer with EXIT before writing the kernel: all-zero SASS is an
 * illegal instruction and the SM prefetches past the end of a program. */
static void write_code(gaia_buffer *code, const hp_word *prog, unsigned count) {
  hp_word pad = hp_exit(hp_ctrl_safe());
  hp_word *slot = (hp_word *)code->hostPtr;
  for (unsigned i = 0; i < code->size / sizeof(hp_word); i++) slot[i] = pad;
  memcpy(code->hostPtr, prog, count * sizeof(hp_word));
}

/* Kernel parameters, in constant bank 0 in CUDA's layout. */
static void write_params(pr_buffers *b, NvU32 blockX, const pr_kernel *k) {
  volatile NvU8 *cb = (volatile NvU8 *)b->scratch.hostPtr;
  *(volatile NvU32 *)(cb + HERMES_CBUF0_NTID_X) = blockX;
  *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM_N(0)) = b->out.gpuAddr;
  *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM_N(1)) = b->in.gpuAddr;
  *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM_N(2)) = b->inB.gpuAddr;
  *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM_N(3)) = b->inC.gpuAddr;
  const float s[HERMES_CBUF0_SCALAR_COUNT] = {
      k->scalar, k->scalar2, k->scalar3, k->scalar4, k->scalar5, k->scalar6};
  for (unsigned i = 0; i < HERMES_CBUF0_SCALAR_COUNT; i++)
    *(volatile NvU32 *)(cb + HERMES_CBUF0_SCALAR_N(i)) =
        k->rawScalar[i] ? k->rawScalar[i] : pr_f2u(s[i]);
}

static char g_mut[96];

/* Run one kernel. Returns NULL on success or the reason it failed. */
static const char *run_kernel(aether_device *d, hermes_channel *c,
                              pr_buffers *b, const pr_kernel *k) {
  volatile NvU32 *o = (volatile NvU32 *)b->out.hostPtr;
  for (unsigned i = 0; i < FENCE_OFFSET / 4 + 4; i++) o[i] = 0;
  for (unsigned i = 0; i < PR_N; i++)
    ((volatile NvU32 *)b->inC.hostPtr)[i] = 0;
  if (k->fill)
    k->fill((volatile NvU32 *)b->in.hostPtr, (volatile NvU32 *)b->inB.hostPtr);
  if (k->fillC) k->fillC((volatile NvU32 *)b->inC.hostPtr);
  if (k->seed) k->seed(o);
  write_params(b, k->blockX, k);

  hp_word prog[PR_MAX_INSTRUCTIONS];
  const unsigned count = k->build(prog, b->out.gpuAddr, b->in.gpuAddr);
  write_code(&b->code, prog, count);

  NvU32 qmd[HERMES_QMD_DWORDS];
  hermes_qmd_build(qmd, b->code.gpuAddr, b->scratch.gpuAddr, k->gridX, 1, 1,
                   k->blockX, 1, 1, k->sharedBytes,
                   count * (NvU32)sizeof(hp_word));
  /* Mirror what helios_enqueue does for the device path, so this harness
   * launches a kernel the way production launches it. */
  if (k->regs) hermes_qmd_set_regs(qmd, k->regs);
  memcpy(b->qmd.hostPtr, qmd, HERMES_QMD_BYTES);
  __asm__ __volatile__("sfence" ::: "memory");

  hermes_begin(c);
  hermes_launch(c, b->qmd.gpuAddr);
  hermes_semaphore_release(c, b->out.gpuAddr + FENCE_OFFSET, FENCE_VALUE);
  hermes_submit(d, c);
  hermes_ring(c, (volatile NvU32 *)c->userd.hostPtr, c->doorbell, c->token);

  /* Wait on the kernel's OWN effect, not only the fence: a broken fence must
   * not be able to mask a working kernel. */
  const NvU64 deadline = now_ns() + 2000000000ull;
  while (now_ns() < deadline && k->check(o) != NULL) {}
  const char *why = k->check(o);
  if (why) return why;

  if (k->checkAux) {
    why = k->checkAux((const volatile NvU32 *)b->inB.hostPtr,
                      (const volatile NvU32 *)b->inC.hostPtr);
    if (why) return why;
  }

  /*
   * Test the TEST: a checker that passes everything passes a broken kernel too.
   *
   * Every slot the kernel claims to check is perturbed in turn and the checker
   * must REJECT it. The slot is restored either way, so the output is unchanged
   * by the time anything else looks at it.
   *
   * The perturbation flips an EXPONENT bit, changing the value by a factor of
   * about two. The first version flipped the lowest mantissa bit, on the
   * reasoning that the smallest possible change is the hardest test -- and that
   * was wrong. One unit in the last place is roughly 1.2e-7 in relative terms,
   * comfortably inside the 1e-5 the transcendental checkers allow, and they
   * allow it because MUFU is approximate by design. Nine kernels "failed" this
   * check while behaving exactly as specified.
   *
   * So the mutation has to be larger than any tolerance a checker is entitled
   * to have, and a factor of two is far outside every one here. What this
   * proves is therefore narrower than it first appeared: that the checker LOOKS
   * at each slot it claims to, not that its tolerance is tight. Tightness is
   * argued separately, per checker, next to the tolerance itself.
   */
#define MUTATION_BIT 0x40000000u
  const NvU32 work = k->workElements ? k->workElements : PR_N;
  const NvU32 checked = k->checkedElements ? k->checkedElements : work;
  /*
   * FENCE BETWEEN THE PERTURBATION AND THE READ, because the output buffer is
   * WRITE-COMBINED and this loop is the one place the HOST writes it.
   *
   * chk_zeros compares every element against 0.0f and the mutation sets 2.0f,
   * so it cannot legitimately miss -- yet this check failed intermittently,
   * roughly one gate run in two, at a different index each time (o[62], then
   * o[56]). That pattern is not a checker with a gap; it is a store that has
   * not landed. A write-combining buffer may hold the store past the reload
   * that follows it, so k->check() reads back the original zero, sees nothing
   * wrong, and the loop reports the checker as not looking.
   *
   * Everywhere else in this stack the host writes device-visible memory and
   * fences before handing over (see hermes_launch's callers); this loop wrote
   * and immediately re-read without one. Two fences: after the mutation so
   * check() sees it, and after the restore so the NEXT iteration's baseline is
   * the real value rather than a mutated leftover.
   */
  /*
   * WAIT FOR THE OUTPUT TO STOP MOVING before perturbing any of it.
   *
   * The fence has signalled by here, and that turns out not to mean the
   * kernel's writes are all visible to the host: the device was observed
   * rewriting a slot BETWEEN the perturbation and the checker's read, six runs
   * in twelve of an unchanged binary, at a different index each time. The
   * attribution below proves it — "rewritten by the device mid-check" rather
   * than "the checker does not look".
   *
   * Two identical reads of the whole checked range, separated by a fence, is
   * the cheap way to establish that nothing is still in flight. It is a
   * property the FENCE should give and does not, so this is a workaround with
   * its reason attached rather than a fix: the fence's release semantics are
   * worth a separate look, and until then a gate that fails half the time is
   * not evidence of anything.
   */
  for (unsigned settle = 0; settle < 100; settle++) {
    NvU32 a = 0, b = 0;
    for (NvU32 i = 0; i < checked; i++) a ^= o[i] + i;
    __asm__ __volatile__("sfence" ::: "memory");
    for (NvU32 i = 0; i < checked; i++) b ^= o[i] + i;
    if (a == b) break;
  }

  for (NvU32 i = 0; i < checked; i++) {
    /*
     * RETRY THE SLOT WHILE THE DEVICE IS STILL WRITING IT.
     *
     * Settling the buffer first cut the flake from six runs in twelve to three,
     * and three is still not evidence: a write can always arrive after the last
     * settle read. But the clobber is now DETECTABLE — the mutation is verified
     * visible before the checker runs and re-read after it — so the honest
     * response is to try that slot again rather than to report a verdict about
     * the checker that the evidence does not support.
     *
     * A gap is only reported when the mutation SURVIVED the check and the
     * checker still accepted, which is the only observation that means what the
     * message says. The retry is bounded; exhausting it reports the race.
     */
    unsigned attempt = 0;
    const NvU32 saved = o[i];
    const NvU32 want = saved ^ MUTATION_BIT;
  retry:
    o[i] = want;
    __asm__ __volatile__("sfence" ::: "memory");
    /*
     * READ THE MUTATION BACK BEFORE BLAMING THE CHECKER.
     *
     * The fence above cut this flake from one run in two to one in twelve, and
     * one in twelve is still a gate that is not evidence. The failure mode it
     * leaves is indistinguishable from a real one: if the store has not landed,
     * check() reads the ORIGINAL value, finds nothing wrong, and the loop
     * reports "the checker does not look at o[i]" — which is a true statement
     * about a perturbation that never happened.
     *
     * So verify. A bounded spin costs nothing when the store lands (it always
     * does within a few reads) and turns the remaining cases into an honest
     * message about the memory system rather than a false accusation against
     * the kernel's checker.
     */
    unsigned spins = 0;
    while (o[i] != want && spins++ < 1000)
      __asm__ __volatile__("pause" ::: "memory");
    if (o[i] != want) {
      o[i] = saved;
      __asm__ __volatile__("sfence" ::: "memory");
      snprintf(g_mut, sizeof g_mut,
               "o[%u] perturbation never visible: the mapping", i);
      return g_mut;
    }
    const char *caught = k->check(o);
    /*
     * IF THE CHECKER SAW NOTHING, ASK WHETHER THE PERTURBATION WAS STILL THERE.
     *
     * The mutation is verified visible before check() runs, so a store that did
     * not land is ruled out. What is NOT ruled out is the GPU: if the kernel has
     * not retired, it writes its own value back over the mutation between the
     * verify and the checker's read, and the checker is then telling the truth
     * about a buffer that no longer holds a perturbation.
     *
     * That is the whole of this flake — six runs in twelve of an unchanged
     * binary, at a different index each time, which is a race and not a gap.
     * Re-reading the slot separates the two: still mutated means the checker
     * really does not look, restored means the launch was still in flight.
     */
    const int stillMutated = (o[i] == want);
    o[i] = saved;
    __asm__ __volatile__("sfence" ::: "memory");
    if (!caught && !stillMutated) {
      if (++attempt < 200) {
        for (volatile unsigned d = 0; d < 2000; d++) { }
        goto retry;
      }
      snprintf(g_mut, sizeof g_mut,
               "o[%u] rewritten by the device mid-check: not retired", i);
      return g_mut;
    }
    if (!caught) {
      snprintf(g_mut, sizeof g_mut,
               "%s: checker accepts a perturbed o[%u] -- it does not check it",
               k->name, i);
      return g_mut;
    }
  }
  return NULL;
}

static void test_registry(void) {
  HT_CASE("every registered kernel runs correctly");

  aether_device d;
  if (aether_device_open(&d, 0) != 0) {
    if (d.failStage == NULL) {
      printf("skip (no NVIDIA driver)\n");
      ht_case_failed = 0;
      return;
    }
    HT_FAIL("device open failed at %s", d.failStage);
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

  pr_buffers b;
  if ((rc = alloc_buffers(&d, &b)) != 0) {
    HT_FAIL("buffers: %s", aether_status_name((unsigned)rc));
    goto done;
  }
  init_engine(&d, &c, &b);

  unsigned n = 0, passed = 0;
  const pr_kernel *ks = pr_kernels(&n);
  for (unsigned i = 0; i < n; i++) {
    const char *why = run_kernel(&d, &c, &b, &ks[i]);
    if (why) {
      HT_FAIL("%s: %s [errnotif=%08x]", ks[i].name, why,
              ((NvU32 *)c.errnotif.hostPtr)[2]);
    } else {
      passed++;
      printf("\n        %2u. %-32s ok", i + 1, ks[i].name);
    }
  }
  printf("\n      %u/%u kernels ", passed, n);
  HT_EQ_U64(passed, n);

done:
  free_buffers(&d, &b);
  hermes_channel_close(&d, &c);
  aether_device_close(&d);
  HT_END();
}

void pr_hardware_tests(void) { test_registry(); }
