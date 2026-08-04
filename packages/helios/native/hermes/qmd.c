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
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A 0x02e4
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_A 0x0790
#define NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A 0x07b0

/*
 * The descriptor is BUILT FROM ZERO, not copied.
 *
 * The first version started from a QMD captured off a running CUDA process and
 * overwrote the fields whose meaning had been established. That is a reasonable
 * instinct and it was wrong in a specific way: it carries every word whose
 * meaning had NOT been established, including a driver's private state, its
 * resource claims, and -- twice, caught only by reading a hex dump -- its
 * addresses. A descriptor assembled that way cannot be reasoned about, because
 * no one can say what most of it means.
 *
 * Mesa's NVK driver, which dispatches compute on this exact hardware, starts
 * from an all-zero 64-dword block and sets twelve named fields. Nothing else.
 * That is the model here: every bit in this descriptor is either set on purpose
 * with a name and a citation, or it is zero.
 *
 * Field positions are from NVIDIA's clc7c0qmd.h (open-gpu-doc,
 * classes/compute), which states them as bit ranges over the whole QMD -- so
 * word = lo/32 and the shift is lo%32.
 *
 * THE POSITIONS ARE CHECKED AGAINST THAT HEADER BY A TEST, not by eye. The
 * first transcription got four of nineteen wrong, and two were not typos but
 * assumptions: CTA_RASTER_HEIGHT and CTA_RASTER_DEPTH are SIXTEEN bits, not
 * thirty-two like CTA_RASTER_WIDTH. Writing a 32-bit value into them spills
 * zeroes across whatever occupies the next sixteen bits -- a silent corruption
 * of neighbouring fields, from an assumption of symmetry the hardware does not
 * share.
 */

/* A bit range within the descriptor, named as the header names it. */
typedef struct {
  unsigned lo, hi;
} qmd_field;

/* NVC7C0_QMDV03_00_*, MW(hi:lo) */
static const qmd_field QMD_MAJOR_VERSION = {580, 583};
static const qmd_field QMD_VERSION = {576, 579};
static const qmd_field API_VISIBLE_CALL_LIMIT = {378, 378};
static const qmd_field SAMPLER_INDEX = {382, 382};
static const qmd_field SM_GLOBAL_CACHING_ENABLE = {134, 134};
static const qmd_field CTA_RASTER_WIDTH = {384, 415};
static const qmd_field CTA_RASTER_HEIGHT = {416, 431};
static const qmd_field CTA_RASTER_DEPTH = {448, 463};
static const qmd_field CTA_THREAD_DIMENSION0 = {592, 607};
static const qmd_field CTA_THREAD_DIMENSION1 = {608, 623};
static const qmd_field CTA_THREAD_DIMENSION2 = {624, 639};
static const qmd_field REGISTER_COUNT_V = {648, 656};
static const qmd_field BARRIER_COUNT = {763, 767};
static const qmd_field SHARED_MEMORY_SIZE = {544, 561};
static const qmd_field SHADER_LOCAL_MEMORY_LOW_SIZE = {736, 759};
static const qmd_field SHADER_LOCAL_MEMORY_HIGH_SIZE = {1600, 1623};
static const qmd_field PROGRAM_ADDRESS_LOWER = {1536, 1567};
static const qmd_field PROGRAM_ADDRESS_UPPER = {1568, 1584};
static const qmd_field SASS_VERSION = {1656, 1663};
/* The six cache invalidates, MW(191:186) -- a contiguous run, so one field.
 * NVK sets all of them on every dispatch. The from-zero rebuild set none, which
 * was a regression: the captured skeleton it replaced at least carried four. */
static const qmd_field INVALIDATE_CACHES = {186, 191};
static const qmd_field MIN_SM_CONFIG_SHARED_MEM_SIZE = {562, 567};
static const qmd_field MAX_SM_CONFIG_SHARED_MEM_SIZE = {569, 574};
static const qmd_field TARGET_SM_CONFIG_SHARED_MEM_SIZE = {657, 662};

/*
 * The SM's shared-memory/L1 partition, encoded as (size_kB / 4) + 1.
 *
 * From Mesa's NAK (gv100_smem_size_to_hw) and confirmed against a CUDA capture:
 * the empty kernel's descriptor holds MAX = 26, and (26 - 1) * 4 = 100 KB,
 * which is exactly sm_86's maximum shared memory per SM. MIN and TARGET read 5,
 * i.e. 16 KB.
 *
 * ALL THREE WERE ZERO here, and zero decodes to a partition size that does not
 * exist -- there is no legal SM configuration with 0 KB. A kernel using no
 * shared memory still has to say which partition it wants, because the SM has
 * to be configured before a CTA can be placed on it. That is why a launch of a
 * single EXIT faulted: the fault happens while placing the block, before any
 * instruction is fetched.
 */
#define SMEM_HW(kb) (((kb) / 4u) + 1u)
#define SM86_SMEM_MIN_KB 16u
#define SM86_SMEM_MAX_KB 100u

/* Write `value` into the field. Ranges here never straddle a dword in the
 * fields we set, but the loop handles it anyway rather than relying on that. */
static void qmd_set(NvU32 *qmd, qmd_field f, NvU64 value) {
  for (unsigned b = f.lo; b <= f.hi; b++) {
    const NvU64 bit = (value >> (b - f.lo)) & 1u;
    if (bit)
      qmd[b / 32] |= (NvU32)1u << (b % 32);
    else
      qmd[b / 32] &= ~((NvU32)1u << (b % 32));
  }
}

/* Constant buffers are indexed: CONSTANT_BUFFER_ADDR_LOWER(i) is
 * MW((1055+i*64):(1024+i*64)), _UPPER is MW((1072+i*64):(1056+i*64)),
 * _SIZE_SHIFTED4 is MW((1087+i*64):(1075+i*64)), _VALID(i) is bit 640+i. */
static void qmd_set_cbuf(NvU32 *qmd, unsigned i, NvU64 addr, NvU32 size) {
  const qmd_field lower = {1024 + i * 64, 1055 + i * 64};
  const qmd_field upper = {1056 + i * 64, 1072 + i * 64};
  const qmd_field shifted4 = {1075 + i * 64, 1087 + i * 64};
  const qmd_field valid = {640 + i, 640 + i};
  qmd_set(qmd, lower, addr & 0xffffffffu);
  qmd_set(qmd, upper, (addr >> 32) & 0x1ffffu);
  qmd_set(qmd, shifted4, (size + 15) / 16);
  qmd_set(qmd, valid, 1);
}

void hermes_qmd_build(NvU32 *qmd, NvU64 program, NvU64 scratch, NvU32 gridX,
                      NvU32 gridY, NvU32 gridZ, NvU32 blockX, NvU32 blockY,
                      NvU32 blockZ) {
  (void)scratch; /* no constant buffers: the kernel reads none */
  memset(qmd, 0, HERMES_QMD_BYTES);

  /* Version, and the two enums NVK sets before anything else. */
  qmd_set(qmd, QMD_MAJOR_VERSION, 3);
  qmd_set(qmd, QMD_VERSION, 0);
  qmd_set(qmd, API_VISIBLE_CALL_LIMIT, 1); /* NO_CHECK */
  qmd_set(qmd, SAMPLER_INDEX, 1);          /* INDEPENDENTLY */
  qmd_set(qmd, SM_GLOBAL_CACHING_ENABLE, 1);
  qmd_set(qmd, INVALIDATE_CACHES, 0x3f); /* all six */

  qmd_set(qmd, CTA_RASTER_WIDTH, gridX);
  qmd_set(qmd, CTA_RASTER_HEIGHT, gridY);
  qmd_set(qmd, CTA_RASTER_DEPTH, gridZ);
  qmd_set(qmd, CTA_THREAD_DIMENSION0, blockX);
  qmd_set(qmd, CTA_THREAD_DIMENSION1, blockY);
  qmd_set(qmd, CTA_THREAD_DIMENSION2, blockZ);

  qmd_set(qmd, PROGRAM_ADDRESS_LOWER, program & 0xffffffffu);
  qmd_set(qmd, PROGRAM_ADDRESS_UPPER, (program >> 32) & 0x1ffffu);
  qmd_set(qmd, SASS_VERSION, HERMES_SASS_VERSION_SM86);

  /* Resources. Generous on registers -- 16 is what an empty CUDA kernel asks
   * for, and over-requesting costs occupancy rather than correctness. The rest
   * are genuinely zero for a kernel with no shared memory, no spills and no
   * barriers. */
  qmd_set(qmd, REGISTER_COUNT_V, 16);
  qmd_set(qmd, BARRIER_COUNT, 0);
  qmd_set(qmd, SHARED_MEMORY_SIZE, 0);
  qmd_set(qmd, MIN_SM_CONFIG_SHARED_MEM_SIZE, SMEM_HW(SM86_SMEM_MIN_KB));
  qmd_set(qmd, TARGET_SM_CONFIG_SHARED_MEM_SIZE, SMEM_HW(SM86_SMEM_MIN_KB));
  qmd_set(qmd, MAX_SM_CONFIG_SHARED_MEM_SIZE, SMEM_HW(SM86_SMEM_MAX_KB));
  qmd_set(qmd, SHADER_LOCAL_MEMORY_LOW_SIZE, 0);
  qmd_set(qmd, SHADER_LOCAL_MEMORY_HIGH_SIZE, 0);
}

void hermes_qmd_set_cbuf(NvU32 *qmd, unsigned index, NvU64 addr, NvU32 size) {
  qmd_set_cbuf(qmd, index, addr, size);
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

  /* _A is the HIGH half and _B the low half. */
  hermes_method(c, subchannel, NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2);
  hermes_data(c, (NvU32)(cfg->sharedWindow >> 32));
  hermes_data(c, (NvU32)(cfg->sharedWindow & 0xffffffffu));

  /* Local memory: where it lives, how much there is, and where the aperture
   * sits. NOTE that a CUDA process does NOT emit these on the compute
   * subchannel -- omitting them was tried and behaves identically -- but a
   * kernel that spills needs the backing, so they stay. */
  hermes_method(c, subchannel, NVC7C0_SET_SHADER_LOCAL_MEMORY_A, 2);
  hermes_data(c, (NvU32)((cfg->localMem >> 32) & 0x1ffffu));
  hermes_data(c, (NvU32)(cfg->localMem & 0xffffffffu));

  hermes_method(c, subchannel, NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 2);
  hermes_data(c, (NvU32)((cfg->localWindow >> 32) & 0x1ffffu));
  hermes_data(c, (NvU32)(cfg->localWindow & 0xffffffffu));

  hermes_method(c, subchannel, NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 3);
  hermes_data(c, (NvU32)((cfg->localMemSize >> 32) & 0xffu));
  hermes_data(c, (NvU32)(cfg->localMemSize & 0xffffffffu));
  hermes_data(c, cfg->smCount & 0x1ffu);

  hermes_method(c, subchannel, NVC7C0_SET_SPA_VERSION, 1);
  hermes_data(c, cfg->spaVersion);

  /* 64 reference counters, written high index to low, matching the capture. */
  for (int i = 63; i >= 0; i--) {
    hermes_method(c, subchannel, NVC7C0_SET_CWD_REF_COUNTER, 1);
    hermes_data(c, 0x0008a000u | (NvU32)i);
  }
}

void hermes_launch(hermes_channel *c, NvU64 qmdAddr, const NvU32 *qmd) {
  /* One header, 66 data words: the two address dwords followed by the QMD. */
  hermes_method(c, 1, NVC7C0_SET_INLINE_QMD_ADDRESS_A, 2 + HERMES_QMD_DWORDS);
  hermes_data(c, (NvU32)((qmdAddr >> 8) >> 32));
  hermes_data(c, (NvU32)((qmdAddr >> 8) & 0xffffffffu));
  for (int i = 0; i < HERMES_QMD_DWORDS; i++) hermes_data(c, qmd[i]);
}
