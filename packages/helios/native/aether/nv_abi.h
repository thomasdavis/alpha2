/*
 * nv_abi.h — the vendor ABI, restated in our own terms.
 *
 * WHAT: the scalar types, ioctl escape codes and RM parameter structs needed to
 * talk to the NVIDIA kernel module.
 *
 * WHY: these are an ABI we do not control. Rather than vendor NVIDIA's headers
 * into the tree, we restate exactly the pieces we use and cite each one, so any
 * value can be checked against its source without a checkout. Every struct here
 * has a layout test in aether_test.c — a wrong offset is not a compile error,
 * it is the kernel silently reading the wrong field.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no wrapping, no policy, no allocation.
 * This file is a transcription. Behaviour lives in rm.c and device.c.
 *
 * PROVENANCE: NVIDIA/open-gpu-kernel-modules, tag matching driver 580.159.03.
 *   escape codes  src/nvidia/arch/nvalloc/unix/include/nv_escape.h
 *   parameters    src/common/sdk/nvidia/inc/nvos.h
 *   class ids     src/common/sdk/nvidia/inc/class/cl*.h
 */
#ifndef HELIOS_AETHER_NV_ABI_H
#define HELIOS_AETHER_NV_ABI_H

#include <stddef.h>
#include <stdint.h>

/* --- scalar types -------------------------------------------------------- */
/* nvtypes.h. NvHandle is an opaque 32-bit RM object id; NvP64 is always 64 bits
 * wide even on 32-bit hosts, which is why it is not a plain pointer. */
typedef uint32_t NvHandle;
typedef uint32_t NvU32;
typedef uint64_t NvU64;
typedef int32_t NvV32;
typedef uint64_t NvP64;

/* --- ioctl escape codes -------------------------------------------------- */
/* nv_escape.h. These are the `nr` field of the ioctl request; the full request
 * is assembled by NV_IOWR() in ioctl.h. */
#define NV_ESC_RM_ALLOC_MEMORY 0x27
#define NV_ESC_RM_ALLOC_OBJECT 0x28
#define NV_ESC_RM_FREE 0x29
#define NV_ESC_RM_CONTROL 0x2A
#define NV_ESC_RM_ALLOC 0x2B
#define NV_ESC_RM_DUP_OBJECT 0x34
#define NV_ESC_RM_VID_HEAP_CONTROL 0x4A
#define NV_ESC_RM_MAP_MEMORY 0x4E
#define NV_ESC_RM_UNMAP_MEMORY 0x4F
#define NV_ESC_RM_MAP_MEMORY_DMA 0x57
#define NV_ESC_RM_UNMAP_MEMORY_DMA 0x58

/* --- non-RM escapes -------------------------------------------------------
 *
 * These are NOT in nv_escape.h's RM list; they are defined as NV_IOCTL_BASE + n
 * with NV_IOCTL_BASE = 200, which is why they appear as 0xc8..0xd7 in a trace
 * and match nothing when grepped for as literals.
 *
 * Found by interposing ioctl on a working CUDA process: it issues these before
 * touching RM at all.
 */
#define NV_ESC_CARD_INFO 0xc8          /* base + 0  */
#define NV_ESC_REGISTER_FD 0xc9        /* base + 1  */
#define NV_ESC_ALLOC_OS_EVENT 0xce     /* base + 6  */
#define NV_ESC_CHECK_VERSION_STR 0xd2  /* base + 10 */
#define NV_ESC_ATTACH_GPUS_TO_FD 0xd4  /* base + 12 */
#define NV_ESC_SYS_PARAMS 0xd6         /* base + 14 */
#define NV_ESC_NUMA_INFO 0xd7          /* base + 15 */

/* The character-device major used by /dev/nvidiactl and /dev/nvidiaN.
 * nv-linux.h; only needed if we ever have to mknod the nodes ourselves. */
#define NV_MAJOR_DEVICE_NUMBER 195

/* --- RM object class ids ------------------------------------------------- */
/* One header per class in sdk/nvidia/inc/class/. Only the ones P0 needs. */
#define NV01_ROOT 0x00000000          /* cl0000.h  — the client */
#define NV01_ROOT_CLIENT 0x00000041   /* cl0000.h  — client, user-space variant */
#define NV01_DEVICE_0 0x00000080      /* cl0080.h  — a GPU */
#define NV20_SUBDEVICE_0 0x00002080   /* cl2080.h  — one GPU within a device */
/* CORRECTED against the vendor headers after hardware rejected both values.
 * The originals were transcribed wrong: 0x3d is not a memory class at all
 * (RM answered NV_ERR_INVALID_CLASS), and 0x3e is SYSTEM rather than
 * LOCAL_USER, so a request labelled "vidmem" was really asking for system
 * memory with video-memory attributes and got NV_ERR_INVALID_ARGUMENT.
 *
 * The lesson is about the test, not the constant: aether_test.c asserted these
 * ids against my own transcription, so it agreed with the bug. Values that
 * exist to match an external source have to be checked against that source. */
#define NV01_MEMORY_LOCAL_USER 0x00000040 /* cl0040.h — video memory */
#define NV01_MEMORY_SYSTEM 0x0000003e     /* cl003e.h — system memory */
/* A reserved GPU virtual address range.
 *
 * NOTE the class/struct pairing, which cost an hour: NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS
 * is defined in cl0070.h and therefore belongs to class 0x70, NOT to
 * NV50_MEMORY_VIRTUAL (0x50a0). Pairing that struct with 0x50a0 returns
 * NV_ERR_INVALID_ARGUMENT. Match structs to the header they were declared in. */
#define NV01_MEMORY_VIRTUAL 0x00000070    /* cl0070.h — a reserved VA range */
#define FERMI_VASPACE_A 0x000090f1        /* cl90f1.h — a GPU address space */
#define AMPERE_CHANNEL_GPFIFO_A 0x0000c56f /* clc56f.h — the submission channel */
#define AMPERE_COMPUTE_B 0x0000c7c0        /* clc7c0.h — GA10x compute engine */
#define AMPERE_DMA_COPY_A 0x0000c6b5       /* clc6b5.h — copy engine */

/* --- NVOS21: allocate an object ------------------------------------------ */
/* nvos.h. Used with NV_ESC_RM_ALLOC. hObjectNew is chosen by the caller — RM
 * does not hand out handles, we allocate them from our own space (see rm.c). */
typedef struct {
  NvHandle hRoot;
  NvHandle hObjectParent;
  NvHandle hObjectNew;
  NvV32 hClass;
  NvP64 pAllocParms __attribute__((aligned(8)));
  NvU32 paramsSize;
  NvV32 status;
} NVOS21_PARAMETERS;

/* --- NVOS54: invoke a control command ------------------------------------ */
/* nvos.h. Used with NV_ESC_RM_CONTROL. `cmd` selects among the thousands of
 * NV*_CTRL_CMD_* entry points; `params` points at a per-command struct. */
typedef struct {
  NvHandle hClient;
  NvHandle hObject;
  NvV32 cmd;
  NvU32 flags;
  NvP64 params __attribute__((aligned(8)));
  NvU32 paramsSize;
  NvV32 status;
} NVOS54_PARAMETERS;

/* --- NVOS33: map device memory into our address space --------------------- */
/* nvos.h. Used with NV_ESC_RM_MAP_MEMORY. pLinearAddress is an out-parameter:
 * RM writes the host virtual address it chose. */
typedef struct {
  NvHandle hClient;
  NvHandle hDevice;
  NvHandle hMemory;
  NvU64 offset __attribute__((aligned(8)));
  NvU64 length __attribute__((aligned(8)));
  NvP64 pLinearAddress __attribute__((aligned(8)));
  NvU32 status;
  NvU32 flags;
} NVOS33_PARAMETERS;

/* --- NVOS02: allocate memory --------------------------------------------- */
/* nvos.h. Used with NV_ESC_RM_ALLOC_MEMORY. */
typedef struct {
  NvHandle hRoot;
  NvHandle hObjectParent;
  NvHandle hObjectNew;
  NvV32 hClass;
  NvV32 flags;
  NvP64 pMemory __attribute__((aligned(8)));
  NvU64 limit __attribute__((aligned(8)));
  NvV32 status;
} NVOS02_PARAMETERS;

/* --- status codes -------------------------------------------------------- */
/* nvstatuscodes.h. Only the ones we branch on; the rest are reported numerically
 * by aether_status_name(). */
#define NV_OK 0x00000000
#define NV_ERR_INVALID_ARGUMENT 0x0000001f
#define NV_ERR_INVALID_STATE 0x00000025
#define NV_ERR_NOT_SUPPORTED 0x00000056
#define NV_ERR_INSUFFICIENT_RESOURCES 0x0000001a
#define NV_ERR_NO_MEMORY 0x00000051

#endif /* HELIOS_AETHER_NV_ABI_H */
