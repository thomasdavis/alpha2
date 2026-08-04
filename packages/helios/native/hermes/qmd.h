/*
 * qmd.h — the launch descriptor.
 *
 * WHAT: builds the 64-dword Queue Meta Data block that tells the compute engine
 * what to run: where the program is, how big the grid and blocks are, and how
 * resources are configured.
 *
 * WHY it is a captured skeleton rather than a struct: the Ampere QMD layout
 * (V03_00) is published nowhere. open-gpu-kernel-modules ships cla0c0qmd.h,
 * which is Kepler, and a full CUDA 12.8 install defines it nowhere on disk.
 * Writing a struct would mean inventing field positions from an older
 * generation and hoping -- and "mostly right" is exactly the failure mode that
 * put GP_PUT at the wrong offset and cost days.
 *
 * So the descriptor starts as bytes lifted from a REAL launch on this GPU, and
 * only the fields whose meaning was established by experiment are overwritten.
 * Everything else is carried verbatim: not understood, but known-good.
 *
 * The fields were identified by launching a kernel whose every dimension is a
 * distinct small number no other field would plausibly hold -- grid (3,5,13),
 * block (9,11,7) -- and seeing which dwords contained them. One capture, six
 * fields, no assumptions. Full record and the raw captures:
 * donto-resources/research/alpha-helios-reimagined/AMPERE-QMD-FIELD-MAP.md
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no constant banks, no shared memory, no
 * register-count control, no dependent launches. A first dispatch needs none of
 * them, and every field left as captured is a field that cannot be wrong in a
 * new way.
 */
#ifndef HELIOS_HERMES_QMD_H
#define HELIOS_HERMES_QMD_H

#include "channel.h"

/* The QMD is 64 dwords, and the hardware is told where it is by an address
 * shifted right by 8 — so it must be 256-byte aligned. */
#define HERMES_QMD_DWORDS 64
#define HERMES_QMD_BYTES (HERMES_QMD_DWORDS * 4)
#define HERMES_QMD_ALIGN 256

/*
 * Field positions, established by differential capture. Anything not listed
 * here is deliberately left as captured — see the header comment.
 *
 *   [8]        program address >> 8, low 32 bits
 *   [12..14]   CTA raster width / height / depth   (the grid)
 *   [18] 31:16 CTA thread dimension 0              (block X)
 *   [19] 15:0  CTA thread dimension 1              (block Y)
 *   [19] 31:16 CTA thread dimension 2              (block Z)
 *   [48],[49]  program address, full 64 bits
 */
#define HERMES_QMD_PROGRAM_SHIFTED8 8
#define HERMES_QMD_GRID_X 12
#define HERMES_QMD_GRID_Y 13
#define HERMES_QMD_GRID_Z 14
#define HERMES_QMD_CTA_XY 18
#define HERMES_QMD_CTA_YZ 19
#define HERMES_QMD_PROGRAM_LO 48
#define HERMES_QMD_PROGRAM_HI 49

/*
 * Fill `qmd` (64 dwords) for a launch of `program` over the given geometry.
 *
 * `program` is a GPU virtual address and must be 256-byte aligned, because the
 * shifted-by-8 copy of it in the descriptor cannot represent anything finer.
 */
/*
 * `scratch` backs the four address pairs the descriptor carries. Even a kernel
 * with no parameters and no shared memory has all four set in a real capture,
 * so they are not optional -- but the captured VALUES are pointers into the
 * traced process and must be replaced, not copied.
 *
 * THE REGIONS ARE 64 KiB APART, not 4 KiB, because the descriptor also declares
 * each bank's SIZE and those sizes must fit. Decoding the captured attribute
 * words through CONSTANT_BUFFER_SIZE_SHIFTED4 gives 6400 bytes for bank 0, 2304
 * for bank 1 and 65536 for bank 7 — so packing them 4 KiB apart in a 64 KiB
 * buffer had bank 7 claiming memory that ran off the end of the allocation. The
 * sizes were being copied from the capture while the layout was invented, which
 * is a combination that cannot be right.
 */
void hermes_qmd_build(NvU32 *qmd, NvU64 program, NvU64 scratch, NvU32 gridX,
                      NvU32 gridY, NvU32 gridZ, NvU32 blockX, NvU32 blockY,
                      NvU32 blockZ);

/* Point constant bank `index` at `addr`, marking it valid. Only needed for
 * kernels that read c[index][...] -- a kernel whose addresses are immediates
 * needs no bank at all, and an unset bank is simply invalid rather than wrong. */
void hermes_qmd_set_cbuf(NvU32 *qmd, unsigned index, NvU64 addr, NvU32 size);

/*
 * Constant bank 0 holds the kernel's parameters, and the layout is CUDA's:
 * driver-provided values low, user parameters from 0x160. We follow it because
 * every reference encoding we have -- and therefore every offset baked into an
 * instruction like IMAD.WIDE.U32 Rd, Ra, Rb, c[0x0][0x160] -- assumes it.
 *
 *   0x000   ntid.x   the block width, read when computing a global index
 *   0x160   param 0
 *   0x168   param 1
 *   ...
 */
#define HERMES_CBUF0_NTID_X 0x000
#define HERMES_CBUF0_PARAM0 0x160
/* Three pointers then two scalars: out, a, b, s0, s1. Two because several
 * operations need a pair -- clamp has a low and a high, silu needs log2(e) and
 * 1.0 -- and materialising constants as immediates would mean an FMUL-immediate
 * encoding that has not been verified against ptxas. */
#define HERMES_CBUF0_SCALAR (0x160 + 24)
#define HERMES_CBUF0_SCALAR2 (0x160 + 28)
#define HERMES_CBUF0_BYTES 0x1000

#define HERMES_QMD_SCRATCH_BYTES (256 * 1024)
#define HERMES_QMD_SCRATCH_STRIDE (64 * 1024)

/*
 * Emit the launch into the channel's pushbuffer.
 *
 * The QMD travels INLINE, in the method stream, which is how a working driver
 * does it. Confirmed by capture: SET_INLINE_QMD_ADDRESS_A is emitted with a
 * count of 66, not 2 — 0x318, 0x31c and LOAD_INLINE_QMD_DATA at 0x320 onward
 * are contiguous, so one header carries the two address dwords and all 64 QMD
 * dwords together. (Which is also why searching a trace for
 * LOAD_INLINE_QMD_DATA finds nothing: it never appears as a header.)
 *
 * `qmdAddr` is where the hardware stages the descriptor; it must be 256-byte
 * aligned GPU-visible memory.
 */
void hermes_launch_inline(hermes_channel *c, NvU64 qmdAddr, const NvU32 *qmd);

/* The PCAS launch: the descriptor is already in GPU memory at `qmdAddr` and the
 * hardware is handed only the address. This is what Mesa's NVK emits. */
void hermes_launch(hermes_channel *c, NvU64 qmdAddr);

/*
 * Initialise the compute engine on this channel. Must precede the first launch.
 *
 * WHY this is not optional: without it a launch is accepted, the GPU consumes
 * the pushbuffer, and the SM raises ROBUST_CHANNEL_GR_EXCEPTION (13) --
 * observed exactly that way. The engine has to be told which ISA it is
 * decoding before it is handed instructions.
 *
 * The sequence is the one a CUDA process emits once per channel, captured with
 * tools/qmd_spy.c:
 *
 *   SET_OBJECT          0x0000  the raw class id
 *   (0x0100)            0x0100  zero
 *   SET_SPA_VERSION     0x0310  0x0806 on sm_86 -- major 8, minor 6
 *   SET_CWD_REF_COUNTER 0x0248  64 times, counter index descending
 *
 *   SET_SHADER_SHARED_MEMORY_WINDOW  0x02a0/0x02a4   a 64-bit aperture base
 *
 * The shared-memory window was omitted at first on the reasoning that a kernel
 * using no shared memory does not need one. That reasoning is wrong: the window
 * is an APERTURE BASE for the SM's address decode, not storage, and leaving it
 * zero faults a kernel that never touches shared memory -- a bare EXIT raised
 * GR_EXCEPTION until it was set. The value is ours to choose; it only has to
 * sit somewhere no real mapping does.
 */
/*
 * Everything the compute engine needs configured before a launch.
 *
 * LOCAL MEMORY is here because a kernel that uses none still needs it backed.
 * The SM allocates per-thread local storage at launch, before any instruction
 * retires, so an unbacked engine faults on a kernel consisting of a single EXIT
 * -- which is exactly what was observed. The methods are
 * SET_SHADER_LOCAL_MEMORY_A/B (the backing address),
 * SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A/B/C (its size, and the SM count it is
 * sized for), and SET_SHADER_LOCAL_MEMORY_WINDOW_A/B (the aperture base).
 *
 * Both WINDOWS are aperture bases for address decode, not storage. They are
 * ours to choose and only have to sit where no real mapping does.
 */
typedef struct {
  NvU32 classId;      /* AMPERE_COMPUTE_B — the raw class id */
  NvU32 spaVersion;   /* HERMES_SPA_VERSION_SM86 */
  NvU64 sharedWindow; /* shared-memory aperture base */
  NvU64 localWindow;  /* local-memory aperture base */
  NvU64 localMem;     /* GPU address backing per-thread local memory */
  NvU64 localMemSize; /* bytes of that backing */
  NvU32 smCount;      /* SMs the local memory is sized for */
} hermes_compute_config;

void hermes_compute_init(hermes_channel *c, NvU32 subchannel,
                         const hermes_compute_config *cfg);

/* An aperture base far above anything Gaia hands out (it allocates upward from
 * 0x04000000), so the window cannot overlap a real buffer. CUDA picks a high
 * address of the same shape. */
/*
 * THE WINDOWS ARE NOT OURS TO PICK FREELY.
 *
 * They were first set to invented high addresses (0x7f00_00000000 and
 * 0x7e00_00000000) on the reasoning that a window only has to sit where no real
 * mapping does. Mesa's NVK driver -- which dispatches compute on this exact
 * hardware -- uses specific values for Volta through Ada:
 *
 *     shared window = 0xfe << 24 = 0xFE000000
 *     local  window = 0xff << 24 = 0xFF000000
 *
 * with the comment "reduce likelihood of collision with real buffers by placing
 * the hole at the top of the 4G area". These are slots near the top of the low
 * 4 GiB, not arbitrary 47-bit addresses, and the SM's generic-address decode
 * cares which is which.
 *
 * (NVK also emits SET_PROGRAM_REGION, but only for pre-Volta classes -- Ampere
 * takes an absolute program address, and clc7c0.h has no such method. Worth
 * recording so nobody adds it on the strength of seeing it in that driver.)
 */
#define HERMES_SHARED_WINDOW_DEFAULT 0x00000000fe000000ull
#define HERMES_LOCAL_WINDOW_DEFAULT 0x00000000ff000000ull

/* sm_86: SET_SPA_VERSION carries major 8, minor 6 (fields MINOR 7:0, MAJOR
 * 15:8). SASS_VERSION is the QMD's own field and is a single byte, 0x86 -- both
 * are required and they are not interchangeable. */
/* AMPERE_COMPUTE_B, and the SM count of the part we develop on. Named here so
 * a caller does not have to repeat magic numbers to launch anything. */
#define HERMES_COMPUTE_CLASS 0xc7c0u
#define HERMES_SM_COUNT_SM86 46u

#define HERMES_SPA_VERSION_SM86 0x0806
#define HERMES_SASS_VERSION_SM86 0x86u

/* Bind the compute class to a subchannel. Must precede the first launch.
 * SET_OBJECT takes the RAW class id (0xc7c0) — confirmed by capture, correcting
 * an earlier note that assumed it wanted the classEngineID from
 * NV906F_CTRL_GET_CLASS_ENGINEID. */
void hermes_set_object(hermes_channel *c, NvU32 subchannel, NvU32 classId);

#endif /* HELIOS_HERMES_QMD_H */
