/*
 * stall_probe.c — the shortest stall a dependent pair of fixed-latency
 * instructions can carry and still be right.
 *
 * WHAT: builds a chain of N instructions where each one reads the register the
 * one before it wrote, encodes a chosen stall count on every link, runs it, and
 * compares the result against arithmetic. Sweeps the stall from 15 down to 0
 * and reports the smallest that still gives the right answer.
 *
 * WHY IT EXISTS: hp_ctrl_safe() encodes stall 15, the maximum, on every
 * instruction in every kernel. control.h says why -- there is no scheduler yet,
 * and from Kepler onwards the hardware has no interlock for this, so a stall
 * that is too short reads a stale register and returns a plausible wrong number
 * with no fault. It also says what it costs: "slower than anything the vendor
 * compiler would emit".
 *
 * It costs 30x. A step's 184 kernels take ~32 ms of GPU time against ~4 ms for
 * the same count of a 1024-element add, and the difference is issue latency, not
 * arithmetic. So the constant is worth knowing rather than assuming, and the
 * only honest way to know it is to ask the hardware.
 *
 * WHY A DEPENDENT CHAIN IS THE RIGHT SHAPE: it is the case the stall exists to
 * protect. Independent instructions cannot expose a too-short stall, so a probe
 * built from them would report 0 for everything and be confidently wrong.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: it does not choose the constant. A
 * measured minimum is a boundary, not a setting -- the value that ships carries
 * margin over it, because this probe runs one occupancy on one card and a
 * boundary found by experiment deserves less trust than one found in a manual
 * we do not have.
 */
/*
 * LAYERING: this is a tool, so it may reach down to prometheus and no further.
 * It opens its own device, channel and buffers rather than borrowing
 * helios_context, which lives ABOVE it -- the first version did borrow it and
 * the gate caught the upward include, which is what that rule is for.
 */
#include "../aether/ioctl.h"
#include "../hermes/pushbuffer.h"
#include "../hephaestus/sm86.h"
#include "../prometheus/kernel.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#define CHAIN 64

/* A stall of `s` on every link of the chain. */
static hp_control at(unsigned s) {
  hp_control c = {s, 0, HP_NO_BARRIER, HP_NO_BARRIER, 0, 0};
  return c;
}

/*
 * Overwrite the stall on an ALREADY-ENCODED instruction.
 *
 * Needed because sm86_flow.c now CLAMPS the stall on ISETP to 15, on the
 * strength of the 13 this probe measured. That clamp made the probe blind to
 * the very thing it measured: asking for stall 3 got 15, the answer came back
 * right, and the tool cheerfully reported a minimum of 0 -- an instrument
 * reporting the policy instead of the hardware.
 *
 * So the probe writes the field itself. The stall is the low four bits of the
 * control field, which begins at bit 105, so bits 41..44 of the high word.
 */
static hp_word with_stall(hp_word w, unsigned s) {
  w.hi = (w.hi & ~(0xfULL << 41)) | ((uint64_t)(s & 0xf) << 41);
  return w;
}

/*
 * Each kind is a chain whose value after N links is known by arithmetic.
 *
 * The integer chains add one per link, so the answer is the link count. The
 * float chain multiplies by one and adds one, which is the same count in
 * floating point and stays exact well past 64.
 */
/*
 * IMAD_WIDE and ISETP are here because they are the two forms with a reason to
 * differ from the ALU chain: one writes a 64-bit REGISTER PAIR, the other
 * writes a PREDICATE, and neither travels the path the other four do. A global
 * constant justified only by the four would be an extrapolation across exactly
 * the boundary where extrapolation fails.
 *
 * They cannot be chained -- an address is used once and a predicate guards one
 * instruction -- so they are single dependent PAIRS, and their answer is a
 * sentinel that only appears if the consumer saw the producer's value.
 */
/*
 * MOV_CONST and HADD2 complete the set that a lowered default would govern.
 *
 * MUFU is not here and does not need to be: every emitter pairs it with
 * hp_ctrl_setbar, so its consumers wait on a barrier and its stall decides
 * nothing. HADD2 is here precisely because cast.c does NOT -- the second
 * half_to_float carries hp_ctrl_safe(), so lowering that default moves it.
 */
typedef enum { K_IADD3, K_IMAD, K_FFMA, K_SHF_LOP3, K_IMAD_WIDE, K_ISETP,
               K_MOV_CONST, K_HADD2 } kind;
static const char *kindName[] = {"IADD3", "IMAD", "FFMA", "SHF+LOP3",
                                 "IMAD.WIDE", "ISETP->@P", "MOV c[]", "HADD2"};

static unsigned build(hp_word *p, kind k, unsigned stall) {
  unsigned n = 0;
  enum { R_ACC = 4, R_TMP = 5, R_ADDR = 6, R_ONE = 8 };

  p[n++] = hp_mov_imm(R_ACC, 0, at(15));
  p[n++] = hp_mov_imm(R_ONE, 1, at(15));

  /*
   * The address pair, under test: the store below reads R_ADDR:R_ADDR+1. Too
   * short a stall and it stores through a stale pair, which lands somewhere
   * else or faults -- either way the sentinel we are watching never changes.
   */
  if (k == K_IMAD_WIDE) {
    p[n++] = hp_mov_imm(R_ACC, CHAIN, at(15));
    p[n++] = hp_imad_wide_const(R_ADDR, HP_RZ, HP_RZ, 0, HERMES_CBUF0_PARAM_N(0),
                                at(stall));
    p[n++] = hp_stg(R_ADDR, R_ACC, 0, at(15));
    p[n++] = hp_exit(at(15));
    return n;
  }

  /* The predicate, under test: the guarded add is the only thing that writes
   * the answer, so a stale predicate leaves zero. */
  if (k == K_ISETP) {
    p[n++] = with_stall(hp_isetp_gt_imm(0, R_ONE, 0, at(stall)), stall);
    p[n++] = hp_predicated(hp_iadd3_imm(R_ACC, R_ACC, CHAIN, at(15)), 0, 0);
    p[n++] = hp_imad_wide_const(R_ADDR, HP_RZ, HP_RZ, 0, HERMES_CBUF0_PARAM_N(0),
                                at(15));
    p[n++] = hp_stg(R_ADDR, R_ACC, 0, at(15));
    p[n++] = hp_exit(at(15));
    return n;
  }

  /* The bank read under test: the multiply below is the only source of the
   * answer, so a stale register leaves whatever R_ACC held. */
  if (k == K_MOV_CONST) {
    p[n++] = hp_mov_const(R_ONE, 0, HERMES_CBUF0_SCALAR_N(0), at(stall));
    p[n++] = hp_imad_imm(R_ACC, R_ONE, 1, HP_RZ, at(15));
    p[n++] = hp_imad_wide_const(R_ADDR, HP_RZ, HP_RZ, 0, HERMES_CBUF0_PARAM_N(0),
                                at(15));
    p[n++] = hp_stg(R_ADDR, R_ACC, 0, at(15));
    p[n++] = hp_exit(at(15));
    return n;
  }

  /* half(CHAIN) in the low half of a packed pair, widened back. 64.0 is 0x5400
   * as an IEEE half and exact, so the comparison is equality rather than a
   * tolerance -- a tolerance would accept a stale register holding something
   * close. */
  if (k == K_HADD2) {
    p[n++] = hp_mov_imm(R_TMP, 0x5400u, at(15));
    p[n++] = hp_half_to_float(R_ACC, R_TMP, HP_HALF_LO, at(stall));
    p[n++] = hp_imad_wide_const(R_ADDR, HP_RZ, HP_RZ, 0, HERMES_CBUF0_PARAM_N(0),
                                at(15));
    p[n++] = hp_stg(R_ADDR, R_ACC, 0, at(15));
    p[n++] = hp_exit(at(15));
    return n;
  }

  for (unsigned i = 0; i < CHAIN; i++) {
    switch (k) {
      case K_IADD3: /* acc = acc + 1 */
        p[n++] = hp_iadd3_imm(R_ACC, R_ACC, 1, at(stall));
        break;
      case K_IMAD: /* acc = acc*1 + 1 */
        p[n++] = hp_imad_imm(R_ACC, R_ACC, 1, R_ONE, at(stall));
        break;
      case K_FFMA: /* acc = acc*1 + 1, in floating point */
        if (i == 0) { p[n++] = hp_mov_imm(R_ACC, 0, at(15)); p[n++] = hp_mov_imm(R_ONE, 0x3f800000u, at(15)); }
        p[n++] = hp_ffma(R_ACC, R_ACC, R_ONE, R_ONE, at(stall));
        break;
      case K_SHF_LOP3:
        /* Two different pipes in one dependent chain: shift left by nothing is
         * not available, so shift RIGHT by 0 (identity) then OR in the carry.
         * tmp = acc >> 0; acc = tmp | 0 ... plus a real increment so the answer
         * still counts. LOP3 with 0xfc is a OR b. */
        p[n++] = hp_shr_imm(R_TMP, R_ACC, 0, at(stall));
        p[n++] = hp_lop3(R_ACC, R_TMP, HP_RZ, 0xfc, at(stall));
        p[n++] = hp_iadd3_imm(R_ACC, R_ACC, 1, at(stall));
        break;
      /* The single-pair kinds returned above; naming them keeps -Wswitch
       * useful, so a kind added later cannot be silently unhandled. */
      case K_IMAD_WIDE: case K_ISETP: case K_MOV_CONST: case K_HADD2:
        break;
    }
  }

  /* Store through param 0. The address arithmetic is fully stalled so only the
   * chain under test can be responsible for a wrong answer. */
  p[n++] = hp_imad_wide_const(R_ADDR, HP_RZ, HP_RZ, 0, HERMES_CBUF0_PARAM_N(0), at(15));
  p[n++] = hp_stg(R_ADDR, R_ACC, 0, at(15));
  p[n++] = hp_exit(at(15));
  return n;
}

static NvU64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (NvU64)ts.tv_sec * 1000000000ull + (NvU64)ts.tv_nsec;
}

/* Everything a launch needs, on the layers below this one. */
typedef struct {
  gaia_buffer code, out, cbuf, qmd, lmem;
} probe_buffers;

static int share(aether_device *d, gaia_buffer *b, NvU64 size, gaia_location w) {
  if (gaia_alloc(d, b, size, w) != 0) return -1;
  if (gaia_map_gpu(d, b) != 0) return -1;
  return gaia_map_host(d, b);
}

int main(void) {
  aether_device d;
  if (aether_device_open(&d, 0) != 0) {
    printf("no device (%s)\n", d.failStage ? d.failStage : "?");
    return 1;
  }
  hermes_channel c;
  if (hermes_channel_open(&d, &c) != 0) { printf("no channel\n"); return 1; }

  probe_buffers b;
  memset(&b, 0, sizeof b);
  if (share(&d, &b.out, 4096, GAIA_SYSMEM) != 0 ||
      share(&d, &b.code, 8192, GAIA_SYSMEM) != 0 ||
      share(&d, &b.cbuf, HERMES_CBUF0_BYTES, GAIA_SYSMEM) != 0 ||
      share(&d, &b.qmd, 4096, GAIA_SYSMEM) != 0) { printf("no buffers\n"); return 1; }
  if (gaia_alloc(&d, &b.lmem, 1024 * 1024, GAIA_VIDMEM) != 0 ||
      gaia_map_gpu(&d, &b.lmem) != 0) { printf("no lmem\n"); return 1; }

  /* Engine init, once and in its own submission, exactly as every other user of
   * this channel does -- without it the SM has no shared-memory window. */
  hermes_compute_config cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.classId = HERMES_COMPUTE_CLASS;
  cfg.spaVersion = HERMES_SPA_VERSION_SM86;
  cfg.sharedWindow = HERMES_SHARED_WINDOW_DEFAULT;
  cfg.localWindow = HERMES_LOCAL_WINDOW_DEFAULT;
  cfg.localMem = b.lmem.gpuAddr;
  cfg.localMemSize = b.lmem.size;
  cfg.smCount = HERMES_SM_COUNT_SM86;
  hermes_begin(&c);
  hermes_compute_init(&c, 1, &cfg);
  hermes_submit(&d, &c);
  hermes_ring(&c, (volatile NvU32 *)c.userd.hostPtr, c.doorbell, c.token);

  volatile NvU8 *cb = (volatile NvU8 *)b.cbuf.hostPtr;
  memset((void *)cb, 0, HERMES_CBUF0_BYTES);
  *(volatile NvU32 *)(cb + HERMES_CBUF0_NTID_X) = 1;
  *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM_N(0)) = b.out.gpuAddr;
  *(volatile NvU32 *)(cb + HERMES_CBUF0_SCALAR_N(0)) = CHAIN;

  printf("chain of %d dependent instructions; the answer is %d\n\n", CHAIN, CHAIN);
  printf("%-10s %s\n", "kind", "stall 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0");

  for (int k = K_IADD3; k <= K_HADD2; k++) {
    printf("%-10s      ", kindName[k]);
    int minOk = -1;
    for (int s = 15; s >= 0; s--) {
      hp_word prog[4096];
      const unsigned n = build(prog, (kind)k, (unsigned)s);

      /* EXIT everywhere first: all-zero SASS is illegal and the SM prefetches
       * past the end of a program. */
      hp_word *slot = (hp_word *)b.code.hostPtr;
      const hp_word pad = hp_exit(hp_ctrl_safe());
      for (unsigned i = 0; i < b.code.size / sizeof(hp_word); i++) slot[i] = pad;
      memcpy(slot, prog, n * sizeof(hp_word));

      volatile NvU32 *host = (volatile NvU32 *)b.out.hostPtr;
      *host = 0xdeadbeefu;
      host[64] = 0; /* the fence, well clear of the answer */

      NvU32 qmd[HERMES_QMD_DWORDS];
      hermes_qmd_build(qmd, b.code.gpuAddr, b.cbuf.gpuAddr, 1, 1, 1, 1, 1, 1, 0,
                       n * (NvU32)sizeof(hp_word));
      memcpy(b.qmd.hostPtr, qmd, HERMES_QMD_BYTES);
      __asm__ __volatile__("sfence" ::: "memory");

      hermes_begin(&c);
      hermes_launch(&c, b.qmd.gpuAddr);
      hermes_semaphore_release(&c, b.out.gpuAddr + 256, 0x5eed5eedu);
      hermes_submit(&d, &c);
      hermes_ring(&c, (volatile NvU32 *)c.userd.hostPtr, c.doorbell, c.token);

      const NvU64 deadline = now_ns() + 2000000000ull;
      while (now_ns() < deadline && host[64] != 0x5eed5eedu) {}
      const int done = host[64] == 0x5eed5eedu;

      int ok = 0;
      if (done) {
        if (k == K_FFMA || k == K_HADD2) {
          float f;
          const NvU32 raw = *host;
          memcpy(&f, &raw, 4);
          ok = f == (float)CHAIN;
        } else {
          ok = *host == (NvU32)CHAIN;
        }
      }
      printf("%3s", ok ? "ok" : done ? "X" : "!");
      if (ok) minOk = s; else break;
    }
    printf("   minimum %d\n", minOk);
  }

  gaia_free(&d, &b.lmem);
  gaia_free(&d, &b.qmd);
  gaia_free(&d, &b.cbuf);
  gaia_free(&d, &b.code);
  gaia_free(&d, &b.out);
  hermes_channel_close(&d, &c);
  aether_device_close(&d);
  return 0;
}
