/*
 * tensor.c — see tensor.h.
 */
#include "tensor.h"

#include <string.h>

#define MAX_TENSORS 4096
#define INDEX_BITS 16
#define INDEX_MASK ((1u << INDEX_BITS) - 1u)

/*
 * Size classes are powers of two from 4 KiB up.
 *
 * Rounding up wastes at most half a buffer and it makes the free list an exact
 * match rather than a search: a request either finds a buffer of its class or
 * allocates one. A best-fit search over arbitrary sizes would reuse memory
 * better and would also mean a 1 MiB request could be served by a 64 MiB
 * buffer, quietly holding sixty-three megabytes hostage for the run.
 */
#define MIN_CLASS_SHIFT 12 /* 4 KiB */
#define NUM_CLASSES 20     /* up to 2 GiB */

typedef struct {
  gaia_buffer buf;
  NvU64 requested; /* what the caller asked for, not the class size */
  NvU32 generation;
  int inUse;
  int classIndex;
  int nextFree; /* -1 terminates; links the free list of its class */
} slot;

static slot g_slots[MAX_TENSORS];
static int g_freeHead[NUM_CLASSES];
static unsigned g_used; /* high-water mark of slots ever created */
static helios_tensor_stats g_stats;
static int g_init;

static void init_once(void) {
  if (g_init) return;
  for (int i = 0; i < NUM_CLASSES; i++) g_freeHead[i] = -1;
  g_init = 1;
}

static int class_of(NvU64 bytes) {
  int c = 0;
  NvU64 size = 1ull << MIN_CLASS_SHIFT;
  while (size < bytes && c < NUM_CLASSES - 1) {
    size <<= 1;
    c++;
  }
  return size < bytes ? -1 : c;
}

/*
 * Resolve a handle, or NULL.
 *
 * The generation check is the whole point. Without it a freed handle indexes a
 * slot that has since been handed to someone else, and writing through it
 * corrupts an unrelated tensor -- with no fault, no error, and no way to trace
 * the damage back to the code that caused it.
 */
static slot *resolve(helios_tensor t) {
  if (t == HELIOS_TENSOR_NONE) return NULL;
  const unsigned index = (t & INDEX_MASK) - 1u;
  if (index >= MAX_TENSORS) return NULL;
  slot *s = &g_slots[index];
  if (!s->inUse) return NULL;
  if ((t >> INDEX_BITS) != (s->generation & (0xffffu))) return NULL;
  return s;
}

static helios_tensor make_handle(unsigned index, NvU32 generation) {
  return ((index + 1u) & INDEX_MASK) | ((generation & 0xffffu) << INDEX_BITS);
}

helios_tensor helios_tensor_alloc(helios_context *ctx, NvU64 bytes) {
  init_once();
  if (bytes == 0) return HELIOS_TENSOR_NONE;
  const int c = class_of(bytes);
  if (c < 0) return HELIOS_TENSOR_NONE;

  /* A buffer of this class already held? Take it without touching the driver. */
  if (g_freeHead[c] >= 0) {
    const int index = g_freeHead[c];
    slot *s = &g_slots[index];
    g_freeHead[c] = s->nextFree;
    s->nextFree = -1;
    s->inUse = 1;
    s->requested = bytes;
    g_stats.live++;
    g_stats.pooled--;
    return make_handle((unsigned)index, s->generation);
  }

  if (g_used >= MAX_TENSORS) return HELIOS_TENSOR_NONE;
  slot *s = &g_slots[g_used];
  memset(s, 0, sizeof *s);
  const NvU64 size = 1ull << (MIN_CLASS_SHIFT + c);
  if (gaia_alloc(&ctx->device, &s->buf, size, GAIA_SYSMEM) != 0)
    return HELIOS_TENSOR_NONE;
  if (gaia_map_gpu(&ctx->device, &s->buf) != 0) return HELIOS_TENSOR_NONE;
  if (gaia_map_host(&ctx->device, &s->buf) != 0) return HELIOS_TENSOR_NONE;

  s->classIndex = c;
  s->requested = bytes;
  s->generation = 1;
  s->inUse = 1;
  s->nextFree = -1;
  const unsigned index = g_used++;
  g_stats.live++;
  g_stats.allocations++;
  g_stats.bytesHeld += size;
  return make_handle(index, s->generation);
}

void helios_tensor_free(helios_tensor t) {
  slot *s = resolve(t);
  if (!s) return;
  /* Bumping the generation is what kills every outstanding copy of this
   * handle, not just the one passed in. */
  s->generation++;
  s->inUse = 0;
  const int index = (int)((t & INDEX_MASK) - 1u);
  s->nextFree = g_freeHead[s->classIndex];
  g_freeHead[s->classIndex] = index;
  g_stats.live--;
  g_stats.pooled++;
}

NvU64 helios_tensor_addr(helios_tensor t) {
  const slot *s = resolve(t);
  return s ? s->buf.gpuAddr : 0;
}

void *helios_tensor_host(helios_tensor t) {
  slot *s = resolve(t);
  return s ? s->buf.hostPtr : NULL;
}

NvU64 helios_tensor_bytes(helios_tensor t) {
  const slot *s = resolve(t);
  return s ? s->requested : 0;
}

helios_tensor_stats helios_tensor_get_stats(void) { return g_stats; }

void helios_tensor_release_all(helios_context *ctx) {
  for (unsigned i = 0; i < g_used; i++) gaia_free(&ctx->device, &g_slots[i].buf);
  memset(g_slots, 0, sizeof g_slots);
  memset(&g_stats, 0, sizeof g_stats);
  g_used = 0;
  g_init = 0;
  init_once();
}
