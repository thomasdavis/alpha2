/*
 * memory.h — buffers the GPU and the host can both reach.
 *
 * WHAT: allocate memory, map it into the GPU's address space, and map it into
 * ours.
 *
 * WHY three separate steps: on NVIDIA these really are three things, and
 * conflating them is the first thing that goes wrong when writing a driver.
 * Allocating gives you a memory OBJECT with no address at all. Mapping it into
 * the GPU virtual address space gives you a pointer the shader can dereference.
 * Mapping it into the host address space gives you a pointer the CPU can
 * dereference. Neither address is derivable from the other, and a buffer can
 * usefully have one without the other.
 *
 * Vulkan hid this behind vkAllocateMemory + vkMapMemory + vkGetBufferDeviceAddress;
 * this is the same three operations with the seams showing.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no suballocation, no pooling, no
 * lifetime tracking beyond a free function. The current Helios allocator does
 * all of that and will need to again, but pooling an allocation path that has
 * not yet been proven correct would only make the first bug harder to find.
 *
 * PROVENANCE: NV_MEMORY_ALLOCATION_PARAMS and the NVOS32_ATTR_* bit-fields from
 * sdk/nvidia/inc/nvos.h; NVOS46 (map-to-GPU) and NVOS33 (map-to-host) likewise.
 */
#ifndef HELIOS_GAIA_MEMORY_H
#define HELIOS_GAIA_MEMORY_H

#include "../aether/device.h"

typedef struct {
  NvHandle handle; /* the RM memory object */
  NvU64 size;
  NvU64 gpuAddr;   /* GPU virtual address, 0 until mapped */
  void *hostPtr;   /* host virtual address, NULL until mapped */
  int hostFd;      /* the fd the host mapping was made through; -1 if none */
  NvHandle vaHandle; /* the reserved VA range backing gpuAddr; 0 if none */
  int location;      /* GAIA_SYSMEM or GAIA_VIDMEM — decides the mapping fd */
  int cached;        /* host-cacheable rather than write-combined; decides the
                      * CACHING_TYPE the host mapping asks for */
  NvU32 mapFlags;    /* override for gaia_map_host; 0 means "use the default
                      * for this location". Registers need their own. */
  /*
   * Chunks 1..n-1 of a gaia_alloc_large buffer. Chunk 0 is `handle`, so a
   * single-chunk buffer leaves this empty and behaves exactly as before.
   * Capped rather than grown: 64 chunks at 4 MiB is 256 MiB, which is far past
   * any single tensor this stack allocates, and a fixed array keeps gaia_free
   * free of ownership questions.
   */
  NvHandle chunks[64];
  unsigned chunkCount;
  /* The stride the chunks were placed at. Stored rather than recomputed from
   * the total: the last chunk is short when the size does not divide, so the
   * average chunk length is NOT the stride and an unmap at the wrong offset
   * silently leaves a mapping behind. */
  NvU64 chunkBytes;
} gaia_buffer;

/* Where the bytes physically live. System memory is host RAM the GPU reaches
 * over PCIe: slower for the GPU, but it is the simplest thing that can work and
 * is the right choice for a first bridge. Device memory is GPU-local. */
typedef enum {
  GAIA_SYSMEM,
  GAIA_VIDMEM,
} gaia_location;

/*
 * Host-mapping flags, per aperture. NVOS33_FLAGS_* from nvos.h:
 *
 *   MAPPING       16:15   DIRECT=1, REFLECTED=2
 *   CACHING_TYPE  25:23   WRITECOMBINED=2, DEFAULT=6
 *
 * Named constants rather than inline literals because a wrong caching mode here
 * does not fail — it silently gives the CPU and the GPU different views of the
 * same bytes. See the long comment in gaia_map_host.
 */
#define GAIA_MAP_FLAGS_SYSMEM ((1u << 15) | (6u << 23)) /* 0x03008000 */
#define GAIA_MAP_FLAGS_VIDMEM ((2u << 15) | (2u << 23)) /* 0x01010000 */

/*
 * A REGISTER window is not memory and must not be mapped like memory.
 *
 * Observed flags for the 64 KiB usermode doorbell page in a working CUDA
 * process: 0x03080002 = ACCESS_WRITE_ONLY(2) | MAPPING_DIRECT(1) |
 * CACHING_TYPE_DEFAULT(6).
 *
 * DIRECT, not REFLECTED. A reflected mapping reaches the register file through
 * a BAR0 window, and it READS correctly -- which is exactly what makes it
 * dangerous here, because reading a ticking GPU clock through one looks like
 * proof that the window is live. Writes are the operation that matters for a
 * doorbell, and they are the operation the two mapping modes disagree about.
 */
#define GAIA_MAP_FLAGS_REGISTERS (2u | (1u << 15) | (6u << 23)) /* 0x03080002 */

/* Allocate with an explicit NVOS32_ATTR word — for buffers whose coherency
 * matters, such as anything the GPU polls. */
int gaia_alloc_attr(aether_device *d, gaia_buffer *b, NvU64 size, NvU32 attr);

/* Allocate `size` bytes. The buffer has no addresses yet. Write-combined. */
int gaia_alloc(aether_device *d, gaia_buffer *b, NvU64 size, gaia_location where);

/*
 * The same, with the host cacheability chosen.
 *
 * `cached` is for memory the HOST READS. Write-combining makes CPU reads bypass
 * the cache — measured at 161x slower than ordinary memory over 2048 floats —
 * which is the correct trade for a pushbuffer and the wrong one for a tensor.
 * See the long comment in gaia_alloc_cached.
 */
int gaia_alloc_cached(aether_device *d, gaia_buffer *b, NvU64 size,
                      gaia_location where, int cached);

/* Map into the GPU virtual address space; fills in b->gpuAddr. */
int gaia_map_gpu(aether_device *d, gaia_buffer *b);

/* Place the buffer at a NAMED address inside a reserved range. RM will not
 * choose an address for us -- mapping without DMA_OFFSET_FIXED fails. */
int gaia_map_gpu_at(aether_device *d, gaia_buffer *b, NvHandle hDma, NvU64 at);

/* Take the next free GPU virtual address range. Bump-only: freed ranges are not
 * recycled, which is honest about being a placeholder and fine while allocation
 * counts are small. */
NvU64 gaia_va_take(NvU64 size);

/* Reserve a GPU virtual address range (NV01_MEMORY_VIRTUAL, class 0x70). This is
 * the object NVOS46 wants in hDma; the VA space itself is rejected with
 * NV_ERR_INVALID_OBJECT_HANDLE. Base must avoid the null page. */
int gaia_reserve_va(aether_device *d, NvHandle *out, NvU64 base, NvU64 size);

/* Map into this process's address space; fills in b->hostPtr. */
int gaia_map_host(aether_device *d, gaia_buffer *b);

/*
 * A buffer LARGER than the kernel will give in one physically contiguous piece.
 *
 * gaia_alloc asks RM for contiguous pages and the kernel's MAX_ORDER stops it
 * at 4 MiB — tools/slab_probe.c walks the sizes and everything from 8 MiB up
 * fails outright. That is not a memory-capacity limit and it was mistaken for
 * one: a 105M model at batch 28 needs a [28,64,640] activation, 4.59 MB in one
 * piece, and it failed on an 8 GiB card holding 4.75 GiB. So the CARD had three
 * gigabytes free and the request was refused anyway.
 *
 * The fix is what a real driver does: reserve one VA range of the full size,
 * then allocate several chunks and place them at consecutive addresses inside
 * it with gaia_map_gpu_at. The GPU sees one contiguous buffer because its MMU
 * says so; nothing physical is contiguous beyond a chunk.
 *
 * `b` describes the whole range: b->gpuAddr is the base and b->size the total.
 * b->handle is the FIRST chunk, and the rest are held internally so gaia_free
 * can release them — which is why this must be freed through gaia_free_large.
 *
 * VIDMEM ONLY and no host mapping, which is the only place it is needed and
 * keeps the chunk bookkeeping out of the host-mapping path.
 */
int gaia_alloc_large(aether_device *d, gaia_buffer *b, NvU64 size,
                     NvU64 chunkBytes);

/* Release a buffer made by gaia_alloc_large, chunks and VA range included. */
void gaia_free_large(aether_device *d, gaia_buffer *b);

/* Release everything the buffer holds. Safe on a partially constructed one. */
void gaia_free(aether_device *d, gaia_buffer *b);

#endif /* HELIOS_GAIA_MEMORY_H */
