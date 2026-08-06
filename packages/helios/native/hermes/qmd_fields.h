/*
 * qmd_fields.h — where every field of the Ampere launch descriptor lives.
 *
 * WHAT: bit positions, transcribed from NVIDIA's own header. Nothing here is
 * inferred and nothing here is code -- it is a table of facts, and it is
 * separate from qmd.c for the same reason isa.h is separate from sm86.h: a
 * constant gets read when checking it against a capture, and the code that uses
 * it gets read when writing a launch. Mixing them means every check wades
 * through logic and every read of the logic wades through bit positions.
 *
 * PROVENANCE: clc7c0qmd.h, NVC7C0_QMDV03_00_*. The MW(x:y) ranges there are
 * inclusive and high-first; these are {low, high}.
 *
 * THE ONE THING TO WATCH: the widths are not uniform. CTA_RASTER_WIDTH is
 * thirty-two bits and CTA_RASTER_HEIGHT is SIXTEEN, sitting in the same dword
 * as DEPTH. Writing a 16-bit field as 32 bits passes every offset check and
 * silently destroys its neighbour, which is why hermes_test asserts widths and
 * not only positions.
 */
#ifndef HERMES_QMD_FIELDS_H
#define HERMES_QMD_FIELDS_H

#include "qmd.h"

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
/*
 * The PREFETCH region -- where the instruction fetcher is allowed to look.
 *
 * These were unset, and everything worked, because a straight-line kernel is
 * fetched sequentially from PROGRAM_ADDRESS and never needs to look anywhere
 * else. The first kernel with a BACKWARD branch faulted the channel with
 * MMU_ERR_FLT on a jump of four instructions, with an instruction stream that
 * disassembles correctly and encodes byte for byte the same as ptxas. A taken
 * forward branch of zero distance worked; one taken backward branch did not.
 * The asymmetry is the tell: forward is streaming, backward is random access
 * into the program, and random access needs the region described.
 *
 * The address fields are SHIFTED by 8 -- the region is 256-byte granular -- and
 * the size is in the same units, nine bits of it, so at most 128 KiB.
 * (clc6c0qmd.h, NVC6C0_QMDV03_00_PROGRAM_PREFETCH_*)
 */
static const qmd_field PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED = {256, 287};
static const qmd_field PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED = {1632, 1640};
static const qmd_field PROGRAM_PREFETCH_SIZE = {1641, 1649};
#define PREFETCH_GRANULE 256u
#define PREFETCH_MAX_UNITS 511u

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

#endif /* HERMES_QMD_FIELDS_H */
