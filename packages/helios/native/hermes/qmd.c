/*
 * qmd.c — see qmd.h.
 */
#include "qmd.h"
#include "pushbuffer.h"

#include <string.h>

/* clc7c0.h */
#define NVC7C0_SET_OBJECT 0x0000
#define NVC7C0_SET_INLINE_QMD_ADDRESS_A 0x0318
#define NVC7C0_SET_SPA_VERSION 0x0310
#define NVC7C0_SET_CWD_REF_COUNTER 0x0248
#define NVC7C0_UNKNOWN_0100 0x0100
#define NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A 0x02a0

/*
 * A real QMD, captured from a CUDA launch on an RTX 3070 (sm_86, driver
 * 580.95.05), with the launch-specific fields zeroed.
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
    /* 16 */ 0x00000000, 0x340c0d00, 0x00000030, 0x00000000,
    /* 20 */ 0x00061083, 0x00000000, 0x00000000, 0x08000000,
    /* 24 */ 0x00000000, 0x00000000, 0x00000006, 0x00000000,
    /* 28 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 32 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 36 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 40 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 44 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 48 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 52 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 56 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    /* 60 */ 0x00000000, 0x00000000, 0x00000000, 0x00000000,
};

void hermes_qmd_build(NvU32 *qmd, NvU64 program, NvU32 gridX, NvU32 gridY,
                      NvU32 gridZ, NvU32 blockX, NvU32 blockY, NvU32 blockZ) {
  memcpy(qmd, QMD_SKELETON, sizeof QMD_SKELETON);

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

void hermes_compute_init(hermes_channel *c, NvU32 subchannel, NvU32 classId,
                         NvU32 spaVersion, NvU64 sharedWindow) {
  hermes_set_object(c, subchannel, classId);

  hermes_method(c, subchannel, NVC7C0_UNKNOWN_0100, 1);
  hermes_data(c, 0);

  /* _A is the HIGH half and _B the low half — the capture reads
   * 0x00007408 then 0xb5000000 for a window at 0x7408_b5000000. */
  hermes_method(c, subchannel, NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2);
  hermes_data(c, (NvU32)(sharedWindow >> 32));
  hermes_data(c, (NvU32)(sharedWindow & 0xffffffffu));

  hermes_method(c, subchannel, NVC7C0_SET_SPA_VERSION, 1);
  hermes_data(c, spaVersion);

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
