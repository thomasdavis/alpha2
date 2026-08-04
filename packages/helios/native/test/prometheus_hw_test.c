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
 * that no kernel's output can reach it. */
#define FENCE_OFFSET 1024
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
  if ((rc = gaia_alloc(d, &b->out, 4096, GAIA_SYSMEM)) != 0) return rc;
  if ((rc = gaia_map_gpu(d, &b->out)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->out)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->in, 4096, GAIA_SYSMEM)) != 0) return rc;
  if ((rc = gaia_map_gpu(d, &b->in)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->in)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->inC, 4096, GAIA_SYSMEM)) != 0) return rc;
  if ((rc = gaia_map_gpu(d, &b->inC)) != 0) return rc;
  if ((rc = gaia_map_host(d, &b->inC)) != 0) return rc;
  if ((rc = gaia_alloc(d, &b->inB, 4096, GAIA_SYSMEM)) != 0) return rc;
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
  for (NvU32 i = 0; i < checked; i++) {
    const NvU32 saved = o[i];
    o[i] = saved ^ MUTATION_BIT;
    const char *caught = k->check(o);
    o[i] = saved;
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
