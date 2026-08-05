/*
 * shared_offset_probe.c — is the LDS/STS immediate offset in BYTES or ELEMENTS?
 *
 * WHAT: writes a known pattern into shared memory with offset zero, then reads
 * it back with a NON-ZERO immediate offset, and reports which interpretation
 * the hardware used.
 *
 * WHY IT EXISTS: hp_lds and hp_sts encode `.X4` scaled addressing — the address
 * register is multiplied by four — and take a 24-bit immediate beside it. Two
 * readings are possible and both are plausible: `reg*4 + imm`, which makes the
 * immediate a BYTE offset, or `(reg + imm)*4`, which makes it an ELEMENT
 * offset. Nothing in this tree depends on the answer today, because every
 * caller passes zero.
 *
 * A tiled matmul does depend on it. Staging four rows of A means addressing
 * shared memory at `r*K + k`, and the natural way to write that is one index
 * register plus a per-row immediate. Guessing wrong there does not fault: it
 * reads a neighbouring row and returns a plausible, wrong dot product — which
 * is precisely what the first tiled attempt did, passing every kernel
 * known-answer and failing a layer suite.
 *
 * So: ask the hardware, once, instead of designing around not knowing. The
 * alternative — folding the row offset into the index register to sidestep the
 * question — costs an instruction per row per iteration on the hottest loop in
 * the model, forever, to avoid a fact that takes one launch to establish.
 *
 * LAYERING: a tool, so it reaches down to prometheus and no further. It opens
 * its own device, channel and buffers rather than borrowing helios_context,
 * which lives above it.
 */
#include "../aether/ioctl.h"
#include "../hermes/pushbuffer.h"
#include "../hephaestus/sm86.h"
#include "../prometheus/kernel.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#define THREADS 8u
/*
 * Four, because it separates the two readings by the widest margin the pattern
 * allows: as a byte offset it lands one element along, as an element offset
 * four. A value of 1 would be ambiguous at the boundary — byte offset 1 is not
 * even element-aligned — and a large one would run off the end.
 */
#define OFFSET 4u

static NvU64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (NvU64)ts.tv_sec * 1000000000ull + (NvU64)ts.tv_nsec;
}

static hp_control safe(void) {
  hp_control c = {15, 0, HP_NO_BARRIER, HP_NO_BARRIER, 0, 0};
  return c;
}

/*
 * shared[tid] = tid*10 ; barrier ; out[tid] = LDS(tid, OFFSET)
 *
 * Thread 0 is the one that answers: it reads shared[1] under the byte reading
 * (10) and shared[4] under the element reading (40). Every other thread agrees,
 * which is what makes a single wrong lane distinguishable from a wrong
 * convention.
 */
static unsigned build(hp_word *p) {
  unsigned n = 0;
  enum { R_TID = 0, R_VAL = 1, R_ESIZE = 2, R_ADDR = 4 /* R4:R5 */ };

  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(0));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(0));

  /* shared[tid] = tid * 10, written at offset zero so the STORE side is not
   * also under test — one unknown at a time. */
  p[n++] = hp_imad_imm(R_VAL, R_TID, 10, HP_RZ, safe());
  p[n++] = hp_sts(R_TID, R_VAL, 0, safe());
  p[n++] = hp_bar_sync(safe());

  /* The question, for shared memory. */
  p[n++] = hp_lds(R_VAL, R_TID, OFFSET, hp_ctrl_setbar(1));

  p[n++] = hp_imad_wide_const(R_ADDR, R_TID, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), safe());
  p[n++] = hp_stg(R_ADDR, R_VAL, 0, hp_ctrl_wait(1));
  p[n++] = hp_exit(safe());
  return n;
}

static int share(aether_device *d, gaia_buffer *b, NvU64 size, gaia_location w) {
  if (gaia_alloc(d, b, size, w) != 0) return -1;
  if (gaia_map_gpu(d, b) != 0) return -1;
  if (gaia_map_host(d, b) != 0) return -1;
  return 0;
}

int main(void) {
  aether_device d;
  hermes_channel c;
  if (aether_device_open(&d, 0) != 0) { printf("no device\n"); return 1; }
  if (hermes_channel_open(&d, &c) != 0) { printf("no channel\n"); return 1; }

  gaia_buffer code, out, cbuf, qmd, lmem;
  if (share(&d, &code, 4096, GAIA_SYSMEM) != 0 ||
      share(&d, &out, 4096, GAIA_SYSMEM) != 0 ||
      share(&d, &cbuf, 65536, GAIA_SYSMEM) != 0 ||
      share(&d, &qmd, 4096, GAIA_SYSMEM) != 0) { printf("no buffers\n"); return 1; }
  if (gaia_alloc(&d, &lmem, 1024 * 1024, GAIA_VIDMEM) != 0 ||
      gaia_map_gpu(&d, &lmem) != 0) { printf("no lmem\n"); return 1; }

  hermes_compute_config cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.classId = HERMES_COMPUTE_CLASS;
  cfg.spaVersion = HERMES_SPA_VERSION_SM86;
  cfg.sharedWindow = HERMES_SHARED_WINDOW_DEFAULT;
  cfg.localWindow = HERMES_LOCAL_WINDOW_DEFAULT;
  cfg.localMem = lmem.gpuAddr;
  cfg.localMemSize = lmem.size;
  cfg.smCount = HERMES_SM_COUNT_SM86;
  hermes_begin(&c);
  hermes_compute_init(&c, 1, &cfg);
  hermes_submit(&d, &c);
  hermes_ring(&c, (volatile NvU32 *)c.userd.hostPtr, c.doorbell, c.token);

  memset(cbuf.hostPtr, 0, 65536);
  *(volatile NvU64 *)((NvU8 *)cbuf.hostPtr + HERMES_CBUF0_PARAM_N(0)) = out.gpuAddr;
  *(volatile NvU32 *)((NvU8 *)cbuf.hostPtr + HERMES_CBUF0_NTID_X) = THREADS;

  hp_word prog[64];
  const unsigned n = build(prog);
  memcpy(code.hostPtr, prog, n * sizeof(hp_word));

  volatile NvU32 *host = (volatile NvU32 *)out.hostPtr;
  for (unsigned i = 0; i < THREADS; i++) host[i] = 0xffffffffu;
  host[64] = 0;

  NvU32 q[HERMES_QMD_DWORDS];
  hermes_qmd_build(q, code.gpuAddr, cbuf.gpuAddr, 1, 1, 1, THREADS, 1, 1,
                   256, n * (NvU32)sizeof(hp_word));
  hermes_qmd_set_cbuf(q, 0, cbuf.gpuAddr, 6400);
  memcpy(qmd.hostPtr, q, HERMES_QMD_BYTES);
  __asm__ __volatile__("sfence" ::: "memory");

  hermes_begin(&c);
  hermes_launch(&c, qmd.gpuAddr);
  hermes_semaphore_release(&c, out.gpuAddr + 256, 0x5eed5eedu);
  hermes_submit(&d, &c);
  hermes_ring(&c, (volatile NvU32 *)c.userd.hostPtr, c.doorbell, c.token);

  const NvU64 deadline = now_ns() + 2000000000ull;
  while (now_ns() < deadline && host[64] != 0x5eed5eedu) {}
  if (host[64] != 0x5eed5eedu) { printf("shared offset: TIMED OUT\n"); return 1; }

  printf("shared offset probe: shared[tid] = tid*10, then LDS(tid, %u)\n", OFFSET);
  for (unsigned i = 0; i < THREADS; i++) printf("  tid %u -> %d\n", i, (int)host[i]);

  /* tid 0 decides: shared[1] is 10 under the byte reading, shared[4] is 40
   * under the element reading. */
  const NvU32 asBytes = (0 + OFFSET / 4u) * 10u;
  const NvU32 asElements = (0 + OFFSET) * 10u;
  if (host[0] == asBytes)
    printf("\nLDS VERDICT: the immediate is a BYTE offset — address is reg*4 + imm\n");
  else if (host[0] == asElements)
    printf("\nLDS VERDICT: the immediate is an ELEMENT offset — address is (reg + imm)*4\n");
  else
    printf("\nLDS VERDICT: neither (%d)\n", (int)host[0]);

  /*
   * AND THE SAME QUESTION FOR GLOBAL MEMORY, because the answer decides whether
   * a K loop can be unrolled cheaply.
   *
   * Unrolling the matmul's inner loop is worth ~2x on issue efficiency — 4
   * FFMAs per 17 instructions against 1 per 9 — but only if the four B loads
   * can share one computed address and differ by an immediate. If the immediate
   * is not a byte offset, each load needs its own IMAD.WIDE and most of the win
   * goes with it.
   *
   * LDG takes a 64-bit register PAIR, not a scaled index, so there is no .X4 to
   * reason about and the immediate is almost certainly bytes — "almost
   * certainly" being the phrase that cost an attempt last time.
   */
  for (unsigned i = 0; i < THREADS; i++) host[i] = 0xffffffffu;
  host[64] = 0;
  {
    unsigned m = 0;
    hp_word g[64];
    enum { G_TID = 0, G_VAL = 1, G_ESIZE = 2, G_ADDR = 4 };
    g[m++] = hp_s2r(G_TID, HP_SR_TID_X, hp_ctrl_setbar(0));
    g[m++] = hp_mov_imm(G_ESIZE, 4, hp_ctrl_wait(0));
    /* Address of out[tid], then read out[tid] back with an immediate of 4. The
     * buffer still holds 0xffffffff everywhere, so first write a pattern. */
    g[m++] = hp_imad_wide_const(G_ADDR, G_TID, G_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), safe());
    g[m++] = hp_imad_imm(G_VAL, G_TID, 10, HP_RZ, safe());
    g[m++] = hp_stg(G_ADDR, G_VAL, 0, safe());
    g[m++] = hp_bar_sync(safe());
    g[m++] = hp_ldg(G_VAL, G_ADDR, OFFSET, hp_ctrl_setbar(1));
    g[m++] = hp_stg(G_ADDR, G_VAL, 0, hp_ctrl_wait(1));
    g[m++] = hp_exit(safe());
    memcpy(code.hostPtr, g, m * sizeof(hp_word));
    hermes_qmd_build(q, code.gpuAddr, cbuf.gpuAddr, 1, 1, 1, THREADS, 1, 1, 256,
                     m * (NvU32)sizeof(hp_word));
    hermes_qmd_set_cbuf(q, 0, cbuf.gpuAddr, 6400);
    memcpy(qmd.hostPtr, q, HERMES_QMD_BYTES);
    __asm__ __volatile__("sfence" ::: "memory");
    hermes_begin(&c);
    hermes_launch(&c, qmd.gpuAddr);
    hermes_semaphore_release(&c, out.gpuAddr + 256, 0x5eed5eedu);
    hermes_submit(&d, &c);
    hermes_ring(&c, (volatile NvU32 *)c.userd.hostPtr, c.doorbell, c.token);
    const NvU64 dl = now_ns() + 2000000000ull;
    while (now_ns() < dl && host[64] != 0x5eed5eedu) {}
    if (host[64] != 0x5eed5eedu) { printf("LDG probe TIMED OUT\n"); return 1; }
    printf("\nglobal offset probe: out[tid] = tid*10, then LDG(&out[tid], %u)\n", OFFSET);
    for (unsigned i = 0; i < 4; i++) printf("  tid %u -> %d\n", i, (int)host[i]);
    if (host[0] == 10) printf("\nLDG VERDICT: BYTE offset — one address serves an unrolled loop\n");
    else if (host[0] == 40) printf("\nLDG VERDICT: ELEMENT offset\n");
    else printf("\nLDG VERDICT: neither (%d)\n", (int)host[0]);
  }
  return 0;
}
