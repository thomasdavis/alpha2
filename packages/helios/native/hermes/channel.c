/*
 * channel.c — see channel.h.
 */
#include "channel.h"
#include "../aether/ioctl.h"

#include <string.h>

/* alloc_channel.h */
#define NV_MAX_SUBDEVICES 8
#define CC_IV_DWORDS 3
#define CC_NONCE_DWORDS 8

typedef struct {
  NvU64 base __attribute__((aligned(8)));
  NvU64 size __attribute__((aligned(8)));
  NvU32 addressSpace;
  NvU32 cacheAttrib;
} NV_MEMORY_DESC_PARAMS;

typedef struct {
  NvHandle hObjectError;
  NvHandle hObjectBuffer;
  NvU64 gpFifoOffset __attribute__((aligned(8)));
  NvU32 gpFifoEntries;
  NvU32 flags;
  NvHandle hContextShare;
  NvHandle hVASpace;
  NvHandle hHandleVASpace;
  NvHandle hUserdMemory[NV_MAX_SUBDEVICES];
  NvU64 userdOffset[NV_MAX_SUBDEVICES] __attribute__((aligned(8)));
  NvU32 engineType;
  NvU32 cid;
  NvU32 subDeviceId;
  NvHandle hObjectEccError;
  NV_MEMORY_DESC_PARAMS instanceMem __attribute__((aligned(8)));
  NV_MEMORY_DESC_PARAMS userdMem __attribute__((aligned(8)));
  NV_MEMORY_DESC_PARAMS ramfcMem __attribute__((aligned(8)));
  NV_MEMORY_DESC_PARAMS mthdbufMem __attribute__((aligned(8)));
  NvHandle hPhysChannelGroup;
  NvU32 internalFlags;
  NV_MEMORY_DESC_PARAMS errorNotifierMem __attribute__((aligned(8)));
  NV_MEMORY_DESC_PARAMS eccErrorNotifierMem __attribute__((aligned(8)));
  NvU32 ProcessID;
  NvU32 SubProcessID;
  NvU32 encryptIv[CC_IV_DWORDS];
  NvU32 decryptIv[CC_IV_DWORDS];
  NvU32 hmacNonce[CC_NONCE_DWORDS];
  NvU32 tpcConfigID;
} NV_CHANNEL_ALLOC_PARAMS;

/* Sizes chosen small: this is a bridge, not a production allocator. A 1024-entry
 * ring and a 64 KiB pushbuffer are far more than a single dispatch needs, and
 * being generous costs one page each. */
#define GPFIFO_ENTRIES 1024
#define GPFIFO_BYTES (GPFIFO_ENTRIES * 8) /* each entry is two 32-bit words */
#define PUSHBUFFER_BYTES (64 * 1024)
#define USERD_BYTES 4096

int hermes_channel_open(aether_device *d, hermes_channel *c) {
  memset(c, 0, sizeof *c);
  int rc;

  /* The ring, the methods, and the doorbell all have to be GPU-visible, so all
   * three go through Gaia rather than being ordinary host allocations.
   *
   * VIDMEM rather than SYSMEM because that is the path proven end to end: the
   * sysmem host mapping returned NV_ERR_INVALID_ARGUMENT and has not been
   * chased down yet. Using the proven path keeps this bring-up honest about
   * what is verified. */
  if ((rc = gaia_alloc(d, &c->gpfifo, GPFIFO_BYTES, GAIA_VIDMEM)) != 0) goto fail;
  if ((rc = gaia_map_gpu(d, &c->gpfifo)) != 0) goto fail;
  if ((rc = gaia_map_host(d, &c->gpfifo)) != 0) goto fail;

  if ((rc = gaia_alloc(d, &c->pushbuffer, PUSHBUFFER_BYTES, GAIA_VIDMEM)) != 0) goto fail;
  if ((rc = gaia_map_gpu(d, &c->pushbuffer)) != 0) goto fail;
  if ((rc = gaia_map_host(d, &c->pushbuffer)) != 0) goto fail;

  if ((rc = gaia_alloc(d, &c->userd, USERD_BYTES, GAIA_VIDMEM)) != 0) goto fail;
  if ((rc = gaia_map_host(d, &c->userd)) != 0) goto fail;

  NV_CHANNEL_ALLOC_PARAMS p;
  memset(&p, 0, sizeof p);
  p.gpFifoOffset = c->gpfifo.gpuAddr;
  p.gpFifoEntries = GPFIFO_ENTRIES;
  p.hVASpace = d->vaspace;
  p.hUserdMemory[0] = c->userd.handle;
  p.userdOffset[0] = 0;
  /* engineType 1 is GRAPHICS/compute on Ampere; compute work runs on the
   * graphics engine rather than a separate one. Probed rather than assumed. */
  p.engineType = 1;

  if ((rc = aether_alloc(d, d->device, &c->handle, AMPERE_CHANNEL_GPFIFO_A,
                         &p, sizeof p)) != 0)
    goto fail;

  /* The compute object is what makes the channel able to run kernels; without
   * it the channel exists but has no engine bound. */
  if ((rc = aether_alloc(d, c->handle, &c->compute, AMPERE_COMPUTE_B, NULL, 0)) != 0)
    goto fail;

  c->gpfifoEntries = GPFIFO_ENTRIES;
  c->push = (NvU32 *)c->pushbuffer.hostPtr;
  return 0;

fail:
  hermes_channel_close(d, c);
  return rc;
}

void hermes_channel_close(aether_device *d, hermes_channel *c) {
  if (c->compute) { aether_free(d, c->compute); c->compute = 0; }
  if (c->handle) { aether_free(d, c->handle); c->handle = 0; }
  gaia_free(d, &c->userd);
  gaia_free(d, &c->pushbuffer);
  gaia_free(d, &c->gpfifo);
  memset(c, 0, sizeof *c);
}
