/*
 * qmd.c — see qmd.h.
 */
#include "qmd.h"
#include "pushbuffer.h"

#include <stdlib.h>
#include <string.h>

/* Instruction-set version the SM is told to decode. clc7c0qmd.h calls this
 * SASS_VERSION; 0x86 is sm_86. Distinct from SET_SPA_VERSION (0x0806), which is
 * a method rather than a QMD field -- both are required and they are not
 * interchangeable. */
#define HERMES_SASS_VERSION_SM86 0x86u

/* Instruction cache prefetch, in the units the capture uses. Copied rather than
 * derived: it is a performance hint, and its unit is not established. */
#define PROGRAM_PREFETCH_SIZE 9u

/* clc7c0.h */
#define NVC7C0_SET_OBJECT 0x0000
#define NVC7C0_SET_INLINE_QMD_ADDRESS_A 0x0318
#define NVC7C0_SET_SPA_VERSION 0x0310
#define NVC7C0_SET_CWD_REF_COUNTER 0x0248
#define NVC7C0_UNKNOWN_0100 0x0100
#define NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A 0x02a0
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A 0x02e4
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_A 0x0790
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A 0x07b0

/*
 * A real QMD, captured from a CUDA launch on an RTX 3070 (sm_86, driver
 * 580.95.05), with the launch-specific fields zeroed.
 *
 * One deliberate change from the capture: word 5 bits 30 and 31
 * (INVALIDATE_INSTRUCTION_CACHE, INVALIDATE_SHADER_CONSTANT_CACHE) are set,
 * making 0x3c000000 into 0xfc000000. CUDA leaves them clear because it uploads
 * code once into a cold cache; we rewrite the same GPU addresses on every run,
 * so stale instruction-cache lines are a real hazard and invalidating is free.
 *
 * The capture is from an EMPTY kernel -- `__global__ void k() {}` launched
 * <<<1,1>>> -- deliberately, because that is as close as CUDA can get to the
 * hand-assembled kernel this will first carry. A skeleton taken from a complex
 * kernel brings that kernel's shared-memory sizing and resource claims with it,
 * and every one of those is a way to be wrong that a matching skeleton simply
 * does not have.
 *
 * Every non-zero word here is carried verbatim from working hardware. Some have
 * known meanings and are overwritten by hermes_qmd_build; the rest are
 * configuration whose encoding is not established, and copying them is the
 * honest option -- the alternative is to invent values for fields we cannot
 * name, which is guessing with extra steps.
 *
 * The words that were observed to differ between two different launches, and so
 * are launch-specific rather than fixed, are zeroed here and set by the builder:
 * [8] and [48],[49] (program address), [12..14] (grid), [18],[19] (block).
 *
 * ALSO ZEROED: every word that carries a FOREIGN ADDRESS. [24],[25] held
 * 0x40800002_0443ff7c, and 0x2_0443ff7c is recognisable -- it appears in the
 * same captured pushbuffer as an OFFSET_OUT destination, so it is a pointer
 * into the traced process's own memory. [50],[51] differ between captures in a
 * way that looks like the low half of another such pointer. Copying a
 * skeleton verbatim is sound for configuration; it is not sound for addresses,
 * because an address that was valid in another process is not merely useless
 * here, it is a fault waiting to be taken. Carrying them produced
 * ROBUST_CHANNEL_GR_EXCEPTION (13) with the launch otherwise accepted.
 *
 * [17] and [20] additionally track shared-memory size. This kernel requests
 * none, so they keep the captured values for a launch that also requested very
 * little. When shared memory is needed their encoding has to be resolved first,
 * and two data points did not determine it -- see AMPERE-QMD-FIELD-MAP.md.
 */
static const NvU32 QMD_SKELETON[HERMES_QMD_DWORDS] = {
    /* 0 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 4 */ 0x0000007f, 0x3c000000, 0x00000000, 0x00000000,
    /* 8 */ 0x00000000, 0x00000000, 0x00000000, 0x44010000,
    /* 12 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 16 */ 0x00000000, 0x34140400, 0x00000030, 0x00000000,
    /* 20 */ 0x000a1083, 0x00000000, 0x00000000, 0x08000000,
    /* 24 */ 0x00000000, 0x00000000, 0x00000006, 0x00000000,
    /* 28 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 32 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 36 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 40 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 44 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 48 */ 0x00000000, 0x00000000, 0x00000640, 0x00000000,
    /* 52 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 56 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 60 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
};

/*
 * The four address pairs, with the attribute halves observed in the capture.
 *
 * Each pair is {low 32 bits of the address} then {attribute << 16 | bits 47:32
 * of the address}. The attribute values are carried verbatim because their
 * encoding is not established; the addresses are replaced because the captured
 * ones point into another process.
 *
 * Zeroing them instead was tried first and is wrong: an empty kernel with no
 * parameters still has all four set, so they are structure rather than payload.
 */
static void set_addr_pair(NvU32 *qmd, int lo, NvU64 addr, NvU32 attr) {
  qmd[lo] = (NvU32)(addr & 0xffffffffu);
  qmd[lo + 1] = (attr << 16) | (NvU32)((addr >> 32) & 0xffffu);
}

void hermes_qmd_build(NvU32 *qmd, NvU64 program, NvU64 scratch, NvU32 gridX,
                      NvU32 gridY, NvU32 gridZ, NvU32 blockX, NvU32 blockY,
                      NvU32 blockZ) {
  memcpy(qmd, QMD_SKELETON, sizeof QMD_SKELETON);

  /*
   * SASS_VERSION and the prefetch descriptor, per NVIDIA's own clc7c0qmd.h
   * (open-gpu-doc, classes/compute). The header states the layout as bit ranges
   * over the whole QMD, so word = lo/32:
   *
   *   SASS_VERSION                        MW(1663:1656)  word 51, bits 31:24
   *   PROGRAM_PREFETCH_TYPE               MW(1651:1650)  word 51, bits 19:18
   *   PROGRAM_PREFETCH_SIZE               MW(1649:1641)  word 51, bits 17:9
   *   PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED MW(1640:1632)  word 51, bits  8:0
   *
   * WHY THIS MATTERS MORE THAN ANYTHING ELSE HERE: word 51 was zeroed earlier
   * on the reasoning that it held a foreign address -- its low bits do, they
   * are the high bits of the program's prefetch address. But its TOP byte is
   * SASS_VERSION, and zeroing it tells the SM to decode an ISA version 0.
   * Every launch since raised GR_EXCEPTION before a single instruction retired.
   *
   * The lesson repeats the one from GP_PUT exactly: a word is not a unit of
   * meaning. Reasoning about "this dword looks like a pointer" is reasoning
   * about the wrong object, because the hardware's fields do not respect dword
   * boundaries and two unrelated things routinely share one.
   */
  qmd[51] = (HERMES_SASS_VERSION_SM86 << 24) |
            ((PROGRAM_PREFETCH_SIZE & 0x1ffu) << 9) |
            (NvU32)(((program >> 8) >> 32) & 0x1ffu);

  set_addr_pair(qmd, 24, scratch + 0 * HERMES_QMD_SCRATCH_STRIDE, 0x4080);
  set_addr_pair(qmd, 32, scratch + 1 * HERMES_QMD_SCRATCH_STRIDE, 0x0c84);
  set_addr_pair(qmd, 34, scratch + 2 * HERMES_QMD_SCRATCH_STRIDE, 0x0480);
  set_addr_pair(qmd, 46, scratch + 3 * HERMES_QMD_SCRATCH_STRIDE, 0x8000);

  /* Two representations of the same address, both present in every capture:
   * a full 64-bit pair and a shifted-by-8 copy. Setting only one is the kind of
   * half-configuration that produces no error and no execution. */
  qmd[HERMES_QMD_PROGRAM_LO] = (NvU32)(program & 0xffffffffu);
  qmd[HERMES_QMD_PROGRAM_HI] = (NvU32)(program >> 32);
  qmd[HERMES_QMD_PROGRAM_SHIFTED8] = (NvU32)((program >> 8) & 0xffffffffu);

  qmd[HERMES_QMD_GRID_X] = gridX;
  qmd[HERMES_QMD_GRID_Y] = gridY;
  qmd[HERMES_QMD_GRID_Z] = gridZ;

  /* Block X sits in the HIGH half of [18]; the low half is 0x0030 in every
   * capture and is preserved from the skeleton. */
  qmd[HERMES_QMD_CTA_XY] =
      (qmd[HERMES_QMD_CTA_XY] & 0x0000ffffu) | ((blockX & 0xffffu) << 16);
  qmd[HERMES_QMD_CTA_YZ] = (blockY & 0xffffu) | ((blockZ & 0xffffu) << 16);
}

void hermes_set_object(hermes_channel *c, NvU32 subchannel, NvU32 classId) {
  hermes_method(c, subchannel, NVC7C0_SET_OBJECT, 1);
  hermes_data(c, classId);
}

void hermes_compute_init(hermes_channel *c, NvU32 subchannel,
                         const hermes_compute_config *cfg) {
  hermes_set_object(c, subchannel, cfg->classId);

  hermes_method(c, subchannel, NVC7C0_UNKNOWN_0100, 1);
  hermes_data(c, 0);

  /* _A is the HIGH half and _B the low half — the capture reads
   * 0x00007408 then 0xb5000000 for a window at 0x7408_b5000000. */
  hermes_method(c, subchannel, NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2);
  hermes_data(c, (NvU32)(cfg->sharedWindow >> 32));
  hermes_data(c, (NvU32)(cfg->sharedWindow & 0xffffffffu));

  /* Local memory: where it lives, how much there is, and where the aperture
   * sits. NOTE that a CUDA process does NOT emit these on the compute
   * subchannel -- omitting them was tried and behaves identically -- but a
   * kernel that spills needs the backing, so they stay. All three are needed — an address with no size is as unusable as a
   * size with no address. ADDRESS_UPPER is 16:0, so the high half is masked to
   * 17 bits rather than passed whole. */
  hermes_method(c, subchannel, NVC7C0_SET_SHADER_LOCAL_MEMORY_A, 2);
  hermes_data(c, (NvU32)((cfg->localMem >> 32) & 0x1ffffu));
  hermes_data(c, (NvU32)(cfg->localMem & 0xffffffffu));

  hermes_method(c, subchannel, NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 2);
  hermes_data(c, (NvU32)((cfg->localWindow >> 32) & 0x1ffffu));
  hermes_data(c, (NvU32)(cfg->localWindow & 0xffffffffu));

  /* NON_THROTTLED_A carries SIZE_UPPER (7:0), _B SIZE_LOWER (31:0), and _C
   * MAX_SM_COUNT (8:0) — three contiguous methods, one header. */
  hermes_method(c, subchannel, NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 3);
  hermes_data(c, (NvU32)((cfg->localMemSize >> 32) & 0xffu));
  hermes_data(c, (NvU32)(cfg->localMemSize & 0xffffffffu));
  hermes_data(c, cfg->smCount & 0x1ffu);

  hermes_method(c, subchannel, NVC7C0_SET_SPA_VERSION, 1);
  hermes_data(c, cfg->spaVersion);

  /* 64 reference counters, written high index to low, matching the capture.
   * The payload is 0x0008a000 | index -- the low byte is the counter and the
   * rest is a fixed field whose meaning is not established. */
  for (int i = 63; i >= 0; i--) {
    hermes_method(c, subchannel, NVC7C0_SET_CWD_REF_COUNTER, 1);
    hermes_data(c, 0x0008a000u | (NvU32)i);
  }
}

void hermes_launch(hermes_channel *c, NvU64 qmdAddr, const NvU32 *qmd) {
  /* One header, 66 data words: the two address dwords followed by the QMD.
   * Subchannel 1 for compute — the captures use it consistently, and the
   * compute object is bound there by hermes_set_object. */
  hermes_method(c, 1, NVC7C0_SET_INLINE_QMD_ADDRESS_A, 2 + HERMES_QMD_DWORDS);
  hermes_data(c, (NvU32)((qmdAddr >> 8) >> 32));
  hermes_data(c, (NvU32)((qmdAddr >> 8) & 0xffffffffu));
  for (int i = 0; i < HERMES_QMD_DWORDS; i++) hermes_data(c, qmd[i]);
}
