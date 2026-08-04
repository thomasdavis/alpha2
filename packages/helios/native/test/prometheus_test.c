/*
 * prometheus_test.c — run every kernel in the registry on real hardware.
 *
 * WHAT: brings up one channel, initialises the compute engine once, then walks
 * the registry launching each kernel and checking its output.
 *
 * WHY one channel for all of them: channel bring-up is the slow part and it is
 * separately tested. Reusing it also exercises something a per-kernel channel
 * would not — that the stack can issue many launches in sequence without state
 * leaking between them, which is where two real bugs have already been found
 * (a pushbuffer rewritten under the GPU, and a barrier-less QMD corrupting the
 * NEXT kernel).
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no timing, no occupancy, no comparison
 * against the Vulkan path. Correctness only.
 */
#include "harness.h"

#include "../aether/ioctl.h"
#include "../hermes/pushbuffer.h"
#include "../prometheus/kernel.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

/* The completion fence lives PAST the output, never inside it. At out+64 it
 * overwrote element 16, and the checker then reported "wrote 16 of 64 slots" —
 * indistinguishable from a kernel that launched one warp. */
#define FENCE_OFFSET 1024
#define FENCE_VALUE 0x5eeeeeedu

static NvU64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (NvU64)ts.tv_sec * 1000000000ull + (NvU64)ts.tv_nsec;
}

/* Everything a launch needs, allocated once. */
typedef struct {
  gaia_buffer code, out, in, inB, qmd, scratch, lmem;
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
static void write_params(pr_buffers *b, NvU32 blockX, float scalar,
                         float scalar2) {
  volatile NvU8 *cb = (volatile NvU8 *)b->scratch.hostPtr;
  *(volatile NvU32 *)(cb + HERMES_CBUF0_NTID_X) = blockX;
  *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM0) = b->out.gpuAddr;
  *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM0 + 8) = b->in.gpuAddr;
  *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM0 + 16) = b->inB.gpuAddr;
  *(volatile NvU32 *)(cb + HERMES_CBUF0_SCALAR) = pr_f2u(scalar);
  *(volatile NvU32 *)(cb + HERMES_CBUF0_SCALAR2) = pr_f2u(scalar2);
}

/* Run one kernel. Returns NULL on success or the reason it failed. */
static const char *run_kernel(aether_device *d, hermes_channel *c,
                              pr_buffers *b, const pr_kernel *k) {
  volatile NvU32 *o = (volatile NvU32 *)b->out.hostPtr;
  for (unsigned i = 0; i < FENCE_OFFSET / 4 + 4; i++) o[i] = 0;
  if (k->fill)
    k->fill((volatile NvU32 *)b->in.hostPtr, (volatile NvU32 *)b->inB.hostPtr);
  if (k->seed) k->seed(o);
  write_params(b, k->blockX, k->scalar, k->scalar2);

  hp_word prog[PR_MAX_INSTRUCTIONS];
  const unsigned count = k->build(prog, b->out.gpuAddr, b->in.gpuAddr);
  write_code(&b->code, prog, count);

  NvU32 qmd[HERMES_QMD_DWORDS];
  hermes_qmd_build(qmd, b->code.gpuAddr, b->scratch.gpuAddr, k->gridX, 1, 1,
                   k->blockX, 1, 1, k->sharedBytes);
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
  return k->check(o);
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

/*
 * The registry itself, checked without a GPU.
 *
 * WHY this exists: the hardware test skips where there is no device, and a
 * suite that skips everything runs no checks — which the harness now reports as
 * a failure, correctly. More usefully, these catch the mistakes that are easy
 * to make while ADDING a kernel: a missing checker, a builder that overruns the
 * instruction buffer, a launch geometry that does not cover the elements the
 * checker will read. All of those would otherwise surface as a confusing
 * hardware failure rather than as what they are.
 */
static void test_registry_is_wellformed(void) {
  HT_CASE("every kernel is completely specified");
  unsigned n = 0;
  const pr_kernel *ks = pr_kernels(&n);
  HT_TRUE(n > 0);

  for (unsigned i = 0; i < n; i++) {
    const pr_kernel *k = &ks[i];
    HT_TRUE(k->name != NULL && k->build != NULL && k->check != NULL);
    HT_TRUE(k->blockX > 0 && k->gridX > 0);

    /* The launch must cover every element a checker inspects, or the test
     * would be comparing against memory no thread ever wrote. */
    HT_EQ_U64(k->blockX * k->gridX, PR_N);

    /*
     * And the builder must fit the buffer the runner gives it.
     *
     * Twice the bound, with a sentinel in the upper half: checking the returned
     * count alone happens after the damage, so it cannot catch an overrun. This
     * can.
     */
    hp_word prog[PR_MAX_INSTRUCTIONS * 2];
    const hp_word sentinel = {0xdeadbeefcafef00dull, 0x0123456789abcdefull};
    for (unsigned s = PR_MAX_INSTRUCTIONS; s < PR_MAX_INSTRUCTIONS * 2; s++)
      prog[s] = sentinel;
    const unsigned count = k->build(prog, 0x1000, 0x2000);
    HT_TRUE(count > 0 && count <= PR_MAX_INSTRUCTIONS);
    for (unsigned s = PR_MAX_INSTRUCTIONS; s < PR_MAX_INSTRUCTIONS * 2; s++)
      if (!hp_word_eq(prog[s], sentinel)) {
        HT_FAIL("%s overran its instruction buffer", k->name);
        break;
      }
  }
  HT_END();
}

/* Every kernel must end in EXIT. A program that runs off its own end is not a
 * kernel, and the padding that makes that survivable is a safety net rather
 * than a licence to rely on it. */
static void test_every_kernel_terminates(void) {
  HT_CASE("every kernel ends in EXIT");
  unsigned n = 0;
  const pr_kernel *ks = pr_kernels(&n);
  const hp_word exit_word = hp_exit(hp_ctrl_safe());
  for (unsigned i = 0; i < n; i++) {
    hp_word prog[PR_MAX_INSTRUCTIONS];
    const unsigned count = ks[i].build(prog, 0x1000, 0x2000);
    HT_TRUE(hp_word_eq(prog[count - 1], exit_word));
  }
  HT_END();
}

void ht_run(void) {
  printf("\nprometheus — kernels\n");
  test_registry_is_wellformed();
  test_every_kernel_terminates();
  test_registry();
}
