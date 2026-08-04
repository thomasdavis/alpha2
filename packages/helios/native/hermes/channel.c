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

/* NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS — nvos.h. A channel cannot be
 * allocated directly under the device on Ampere; it must live inside a Time
 * Slice Group. Allocating one under the device gives NV_ERR_INVALID_ARGUMENT
 * for every engineType, which reads as "bad engine" and is really "wrong
 * parent". */
typedef struct {
  NvHandle hObjectError;
  NvHandle hObjectEccError;
  NvHandle hVASpace;
  NvU32 engineType;
  NvU32 bIsCallingContextVgpuPlugin;
} NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS;

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
/* KEPLER_CHANNEL_GROUP_A — class/cla06c.h. Still the channel-group class on
 * Ampere; there is no AMPERE_CHANNEL_GROUP. */
#define KEPLER_CHANNEL_GROUP_A 0x0000A06C

/* Engine index for graphics/compute. Determined by probe: the channel group
 * allocates with NV_OK at 1 and NV_ERR_INVALID_ARGUMENT at 0, 2 and 3. Compute
 * work runs on the graphics engine rather than a separate one. */
#define HERMES_ENGINE_GRAPHICS 1

#define GPFIFO_ENTRIES 1024
#define GPFIFO_BYTES (GPFIFO_ENTRIES * 8) /* each entry is two 32-bit words */
#define PUSHBUFFER_BYTES (64 * 1024)
#define USERD_BYTES 4096

int hermes_channel_open(aether_device *d, hermes_channel *c) {
  memset(c, 0, sizeof *c);
  int rc;

  /* The ring and the methods must be GPU-visible, so both go through Gaia.
   *
   * VIDMEM rather than SYSMEM because that is the path proven end to end; the
   * sysmem host mapping still returns NV_ERR_INVALID_ARGUMENT and has not been
   * chased down. */
  if ((rc = gaia_alloc(d, &c->gpfifo, GPFIFO_BYTES, GAIA_VIDMEM)) != 0) goto fail;
  if ((rc = gaia_map_gpu(d, &c->gpfifo)) != 0) goto fail;
  if ((rc = gaia_map_host(d, &c->gpfifo)) != 0) goto fail;

  if ((rc = gaia_alloc(d, &c->pushbuffer, PUSHBUFFER_BYTES, GAIA_VIDMEM)) != 0) goto fail;
  if ((rc = gaia_map_gpu(d, &c->pushbuffer)) != 0) goto fail;
  if ((rc = gaia_map_host(d, &c->pushbuffer)) != 0) goto fail;

  /* The channel group owns the address space. */
  {
    NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS g;
    memset(&g, 0, sizeof g);
    g.hVASpace = d->vaspace;
    g.engineType = HERMES_ENGINE_GRAPHICS;
    if ((rc = aether_alloc(d, d->device, &c->group, KEPLER_CHANNEL_GROUP_A,
                           &g, sizeof g)) != 0)
      goto fail;
  }

  /*
   * The channel itself, and the two fields that must be LEFT ZERO.
   *
   * hVASpace: the group already carries it. Setting it here as well returns
   * NV_ERR_INVALID_ARGUMENT -- it reads as a missing field but is a duplicated
   * one.
   *
   * hUserdMemory: RM allocates the USERD doorbell page itself. Supplying our
   * own buffer also fails. Both were set in the first attempt, which is why
   * sweeping engineType found nothing: the engine was never the problem.
   */
  {
    NV_CHANNEL_ALLOC_PARAMS p;
    memset(&p, 0, sizeof p);
    p.gpFifoOffset = c->gpfifo.gpuAddr;
    p.gpFifoEntries = GPFIFO_ENTRIES;
    p.engineType = HERMES_ENGINE_GRAPHICS;
    if ((rc = aether_alloc(d, c->group, &c->handle, AMPERE_CHANNEL_GPFIFO_A,
                           &p, sizeof p)) != 0)
      goto fail;
  }

  /* Binding the compute class is what makes the channel able to run kernels.
   * It takes no allocation parameters. */
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
  if (c->group) { aether_free(d, c->group); c->group = 0; }
  gaia_free(d, &c->pushbuffer);
  gaia_free(d, &c->gpfifo);
  memset(c, 0, sizeof *c);
}
