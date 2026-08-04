/*
 * qmd.c — see qmd.h.
 */
#include "qmd_fields.h"
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
#define NVC7C0_SEND_PCAS_A 0x02b4
#define NVC7C0_SEND_SIGNALING_PCAS2_B 0x02c0
#define PCAS_ACTION_INVALIDATE_COPY_SCHEDULE 0x3

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
                      NvU32 blockZ, NvU32 sharedBytes,
                      NvU32 programBytes) {
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

  {
    /* Round the region out to the granule in BOTH directions: down to the
     * granule the program starts in, and up to cover its last byte. Rounding
     * the address down without lengthening the size would leave the tail of a
     * program outside its own prefetch region. */
    const NvU64 base = program & ~(NvU64)(PREFETCH_GRANULE - 1);
    const NvU64 end = program + programBytes;
    NvU64 units = (end - base + PREFETCH_GRANULE - 1) / PREFETCH_GRANULE;
    if (units < 1) units = 1;
    if (units > PREFETCH_MAX_UNITS) units = PREFETCH_MAX_UNITS;
    qmd_set(qmd, PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED, (base >> 8) & 0xffffffffu);
    qmd_set(qmd, PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, (base >> 40) & 0x1ffu);
    qmd_set(qmd, PROGRAM_PREFETCH_SIZE, units);
  }
  qmd_set(qmd, SASS_VERSION, HERMES_SASS_VERSION_SM86);

  /*
   * Registers per thread.
   *
   * 16 was taken from an empty CUDA kernel and is not enough once kernels do
   * real work: the binary elementwise ones use up to R14 for a second address
   * pair, a second loaded value, a scalar and a temporary, and asking for
   * fewer registers than a kernel touches raises GR_EXCEPTION rather than
   * spilling. Ampere allocates in units of 8, so 32 is the next size that
   * comfortably covers everything here.
   *
   * Over-requesting costs occupancy, not correctness. A register allocator will
   * eventually compute this per kernel; until then a generous fixed number is
   * the right trade, because the failure mode of guessing low is a fault and
   * the failure mode of guessing high is slower.
   */
  qmd_set(qmd, REGISTER_COUNT_V, 32);
  /*
   * ONE BARRIER, always.
   *
   * This was 0, and a kernel that executes BAR.SYNC with no barrier allocated
   * does not fault at the barrier -- it corrupts SM state and the NEXT kernel
   * raises GR_EXCEPTION. That misdirection cost a while: the barrier kernel
   * passed, the kernel after it failed, and every hypothesis went to whatever
   * that next kernel did differently. When the barrier kernel happened to run
   * last, the whole suite passed.
   *
   * The tell was positional rather than technical: the failing kernel changed
   * whenever the ORDER changed, which no property of a kernel can explain.
   */
  qmd_set(qmd, BARRIER_COUNT, 1);
  /* Shared memory is rounded up to 256 bytes, the granularity NVK rounds to
   * before writing this field. A kernel that uses none asks for none. */
  qmd_set(qmd, SHARED_MEMORY_SIZE, (sharedBytes + 0xff) & ~0xffu);
  qmd_set(qmd, MIN_SM_CONFIG_SHARED_MEM_SIZE, SMEM_HW(SM86_SMEM_MIN_KB));
  qmd_set(qmd, TARGET_SM_CONFIG_SHARED_MEM_SIZE, SMEM_HW(SM86_SMEM_MIN_KB));
  qmd_set(qmd, MAX_SM_CONFIG_SHARED_MEM_SIZE, SMEM_HW(SM86_SMEM_MAX_KB));

  /*
   * Constant bank 0, even though this kernel reads no constants.
   *
   * NVK always binds at least one -- its root descriptor -- and a CUDA capture
   * of an EMPTY kernel has banks 0, 1 and 7 valid. Two independent drivers
   * never launching without one is a stronger signal than the observation that
   * our kernel does not need it, so bank 0 gets bound to scratch. The size is
   * the minimum constant-buffer alignment on this hardware.
   */
  if (scratch) qmd_set_cbuf(qmd, 0, scratch, HERMES_CBUF0_BYTES);
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

void hermes_launch_inline(hermes_channel *c, NvU64 qmdAddr, const NvU32 *qmd) {
  /* One header, 66 data words: the two address dwords followed by the QMD. */
  hermes_method(c, 1, NVC7C0_SET_INLINE_QMD_ADDRESS_A, 2 + HERMES_QMD_DWORDS);
  hermes_data(c, (NvU32)((qmdAddr >> 8) >> 32));
  hermes_data(c, (NvU32)((qmdAddr >> 8) & 0xffffffffu));
  for (int i = 0; i < HERMES_QMD_DWORDS; i++) hermes_data(c, qmd[i]);
}

/*
 * The other launch path, and the one Mesa's NVK uses.
 *
 * Rather than carrying the descriptor inline, the caller writes it into GPU
 * memory itself and hands the hardware only an address. Two methods:
 *
 *   SEND_PCAS_A             the QMD address, shifted right by 8
 *   SEND_SIGNALING_PCAS2_B  PCAS_ACTION, as an IMMEDIATE method
 *
 * with action INVALIDATE_COPY_SCHEDULE (3) on Ampere -- NVK selects
 * SEND_SIGNALING_PCAS_B with separate invalidate/schedule flags only for Turing
 * and earlier.
 *
 * Worth having both: the inline path is what CUDA emits and the PCAS path is
 * what NVK emits, so a fault that follows one and not the other says something,
 * and a fault that follows both says the descriptor rather than its delivery.
 */
/*
 * A barrier between dispatches, and it is NOT optional.
 *
 * Consecutive launches on one channel PIPELINE -- they do not serialise. The
 * synchronous design hid that completely, because the host waited on a fence
 * after every launch and the wait was the barrier. Batching two dependent
 * kernels without this produced a copy-then-double that returned the copied
 * value: the doubling read its input before the copy had written it, doubled
 * zero, and then the copy landed on top. A plausible number, and the wrong one.
 *
 * WAIT_FOR_IDLE is the blunt instrument -- it drains the whole pipe rather than
 * expressing which kernel depends on which -- and it is the right one here.
 * What batching buys is removing the HOST round trip per launch, the doorbell
 * and the fence spin; it was never going to buy overlap between kernels that
 * read each other's output. Expressing real dependencies needs the QMD's
 * dependent-launch fields, which is a larger change and wants a measurement
 * saying the drain costs something.
 *
 * (clc7c0.h, NVC7C0_WAIT_FOR_IDLE)
 */
#define NVC7C0_WAIT_FOR_IDLE 0x0110

void hermes_barrier(hermes_channel *c) {
  hermes_method(c, 1, NVC7C0_WAIT_FOR_IDLE, 1);
  hermes_data(c, 0);
}

void hermes_launch(hermes_channel *c, NvU64 qmdAddr) {
  hermes_method(c, 1, NVC7C0_SEND_PCAS_A, 1);
  hermes_data(c, (NvU32)(qmdAddr >> 8));

  /* IMMD_DATA_METHOD: the value rides in the header's count field and no data
   * word follows. clc56f.h SEC_OP 4. */
  *c->push++ = (4u << 29) | ((PCAS_ACTION_INVALIDATE_COPY_SCHEDULE & 0x1fffu) << 16) |
               (1u << 13) | ((NVC7C0_SEND_SIGNALING_PCAS2_B >> 2) & 0xfffu);
}
