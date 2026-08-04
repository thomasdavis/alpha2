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

/* FERMI_CONTEXT_SHARE_A. Allocated UNDER the channel group and referenced by
 * the channel through hContextShare. This is what binds a channel to an address
 * space context; without it the channel allocates and schedules cleanly and
 * then never runs anything.
 *
 * Found by tracing a working Vulkan submit on the same GPU: it allocates 0x9067
 * under the TSG immediately before the channel. Nothing in the channel's own
 * parameter documentation says it is required. */
#define FERMI_CONTEXT_SHARE_A 0x00009067

typedef struct {
  NvHandle hVASpace;
  NvU32 flags;
  NvU32 subctxId;
} NV_CTXSHARE_ALLOCATION_PARAMETERS;

/* Control commands. ctrla06c.h, ctrlc36f.h, ctrl906f.h. */
#define NVA06C_CTRL_CMD_GPFIFO_SCHEDULE 0xa06c0101
#define NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN 0xc36f0108
#define NV906F_CTRL_GET_CLASS_ENGINEID 0x906f0101

/* nvos.h, NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT (1:0) */
#define NV_CTXSHARE_FLAGS_SUBCONTEXT_ASYNC 1

/* Engine index for graphics/compute. Determined by probe: the channel group
 * allocates with NV_OK at 1 and NV_ERR_INVALID_ARGUMENT at 0, 2 and 3. Compute
 * work runs on the graphics engine rather than a separate one. */
#define HERMES_ENGINE_GRAPHICS 1

/* A working Vulkan driver requests 0x8000 entries. Matching it removes one more
 * difference from the reference trace. */
#define GPFIFO_ENTRIES 0x8000
#define GPFIFO_BYTES (GPFIFO_ENTRIES * 8) /* each entry is two 32-bit words */
#define PUSHBUFFER_BYTES (64 * 1024)
#define USERD_BYTES 4096

/* Every failure path names itself. See hermes_channel.failStage. */
#define FAIL(stage)                                                            \
  do {                                                                         \
    c->failStage = (stage);                                                    \
    c->failStatus = rc;                                                        \
    goto fail;                                                                 \
  } while (0)

int hermes_channel_open(aether_device *d, hermes_channel *c) {
  memset(c, 0, sizeof *c);
  int rc = 0;

  /* The ring and the methods must be GPU-visible, so both go through Gaia.
   *
   * SYSMEM, which is where drivers conventionally put these: the host writes
   * them constantly and the GPU's fetch engine reads them. Video memory was
   * used until the sysmem host mapping was fixed -- it needed /dev/nvidiactl
   * rather than the device node. */
  if ((rc = gaia_alloc(d, &c->gpfifo, GPFIFO_BYTES, GAIA_SYSMEM)) != 0) FAIL("gpfifo alloc");
  if ((rc = gaia_map_gpu(d, &c->gpfifo)) != 0) FAIL("gpfifo map_gpu");
  if ((rc = gaia_map_host(d, &c->gpfifo)) != 0) FAIL("gpfifo map_host");

  if ((rc = gaia_alloc(d, &c->pushbuffer, PUSHBUFFER_BYTES, GAIA_SYSMEM)) != 0) FAIL("pushbuffer alloc");
  if ((rc = gaia_map_gpu(d, &c->pushbuffer)) != 0) FAIL("pushbuffer map_gpu");
  if ((rc = gaia_map_host(d, &c->pushbuffer)) != 0) FAIL("pushbuffer map_host");

  /* Our own USERD. RM will allocate one itself, but then the only way to reach
   * GP_PUT is to map the channel object -- and a working driver never does
   * that, so the offset semantics of such a mapping are unverified. Supplying
   * our own buffer removes the question: USERD is memory we allocated and
   * mapped, so its field offsets are the ones dev_ram.h documents.
   *
   * RM accepts this only when hVASpace is left zero on the channel. */

  /* The error notifier. A working Vulkan driver passes a non-zero hObjectError
   * on the channel; we had been leaving it zero, which is also why a fault
   * produced no signal anywhere. */
  if ((rc = gaia_alloc(d, &c->errnotif, 4096, GAIA_SYSMEM)) != 0) FAIL("errnotif alloc");
  if ((rc = gaia_map_host(d, &c->errnotif)) != 0) FAIL("errnotif map_host");
  memset(c->errnotif.hostPtr, 0, 4096);

  /* The channel group owns the address space. */
  {
    NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS g;
    memset(&g, 0, sizeof g);
    /* hVASpace ZERO on the group. A working CUDA process passes
     *   hObjectError=0 hObjectEccError=0 hVASpace=0 engineType=1
     * -- the group takes the DEVICE's address space, and naming one explicitly
     * is not what a working driver does. */
    g.engineType = HERMES_ENGINE_GRAPHICS;
    if ((rc = aether_alloc(d, d->device, &c->group, KEPLER_CHANNEL_GROUP_A,
                           &g, sizeof g)) != 0)
      FAIL("KEPLER_CHANNEL_GROUP_A");
  }

  /* The context share, which the channel will reference. */
  {
    NV_CTXSHARE_ALLOCATION_PARAMETERS cs;
    memset(&cs, 0, sizeof cs);
    cs.hVASpace = d->vaspace;
    /* SUBCONTEXT_ASYNC (nvos.h, NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT 1:0).
     * CUDA passes 1 and gets subctxId 0x3f back. The context share -- not the
     * group and not the channel -- is what carries the address space; the group
     * and channel both leave hVASpace zero. */
    cs.flags = NV_CTXSHARE_FLAGS_SUBCONTEXT_ASYNC;
    if ((rc = aether_alloc(d, c->group, &c->ctxshare, FERMI_CONTEXT_SHARE_A,
                           &cs, sizeof cs)) != 0)
      FAIL("FERMI_CONTEXT_SHARE_A");
  }

  /*
   * The channel itself, and the two fields that must be LEFT ZERO.
   *
   * hVASpace: the group already carries it. Setting it here as well returns
   * NV_ERR_INVALID_ARGUMENT -- it reads as a missing field but is a duplicated
   * one.
   *
   * hUserdMemory: RM will allocate USERD itself, but it also accepts ours --
   * PROVIDED hVASpace above is zero. The first sweep set both at once, which is
   * why it concluded neither worked and why sweeping engineType afterwards
   * found nothing: the engine was never the problem.
   */
  {
    NV_CHANNEL_ALLOC_PARAMS p;
    memset(&p, 0, sizeof p);
    p.gpFifoOffset = c->gpfifo.gpuAddr;
    p.gpFifoEntries = GPFIFO_ENTRIES;
    /* engineType ZERO on the CHANNEL, 1 on the GROUP. Observed in a CUDA trace
     * and it makes sense once seen: the TSG owns the engine and the channel
     * inherits it. We had been setting 1 in both places, which is why sweeping
     * this field on the channel changed nothing -- every value including the
     * "right" one was wrong, because the field should not have been set. */
    p.engineType = 0;
    p.hContextShare = c->ctxshare;
    p.hObjectError = c->errnotif.handle;
    if ((rc = aether_alloc(d, c->group, &c->handle, AMPERE_CHANNEL_GPFIFO_A,
                           &p, sizeof p)) != 0)
      FAIL("AMPERE_CHANNEL_GPFIFO_A");
  }

  /* Binding the compute class is what makes the channel able to run kernels.
   * It takes no allocation parameters. */
  if ((rc = aether_alloc(d, c->handle, &c->compute, AMPERE_COMPUTE_B, NULL, 0)) != 0)
    FAIL("AMPERE_COMPUTE_B");

  /*
   * The doorbell window.
   *
   * AMPERE_USERMODE_A is not memory -- it is a 64 KiB MMIO aperture over the
   * register range NV_VIRTUAL_FUNCTION (0x30000..0x3ffff, dev_vm.h). It maps
   * through the ordinary memory-mapping path even so, because as far as RM is
   * concerned it is an object with a BAR mapping.
   */
  if ((rc = aether_alloc(d, d->subdevice, &c->usermode.handle,
                         HERMES_USERMODE_CLASS, NULL, 0)) != 0)
    FAIL("AMPERE_USERMODE_A");
  c->usermode.size = HERMES_USERMODE_BYTES;
  c->usermode.location = GAIA_VIDMEM; /* a device-node mapping, as for any BAR */
  c->usermode.mapFlags = GAIA_MAP_FLAGS_REGISTERS; /* DIRECT, not REFLECTED */
  if ((rc = gaia_map_host(d, &c->usermode)) != 0) FAIL("usermode map_host");
  c->doorbell = (volatile NvU32 *)c->usermode.hostPtr;

  /*
   * ONE control call schedules the channel, and it is on the GROUP.
   *
   * A CUDA trace of nine channels issues NVA06C_CTRL_CMD_GPFIFO_SCHEDULE
   * (0xa06c0101, psz=3) exactly once, on the TSG, after every channel exists.
   * It never calls NVA06F_CTRL_CMD_BIND, NVA06C_CTRL_CMD_BIND, or the channel's
   * own GPFIFO_SCHEDULE.
   *
   * We had been calling all four. Each returned NV_OK, which is precisely why
   * they survived: a redundant call that succeeds looks like diligence. Binding
   * the CHANNEL to engine 1 in particular contradicts leaving its engineType at
   * zero -- the TSG owns the engine, and saying so twice in two different ways
   * is not the same as saying it once.
   *
   * The three-byte size is real: NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS is a typedef
   * of the CHANNEL's NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS, so it carries three
   * NvBools rather than the two the group header suggests. The trace confirms
   * psz=3.
   */
  {
    struct { NvU8 bEnable, bSkipSubmit, bSkipEnable; } sched = { 1, 0, 0 };
    if ((rc = aether_control(d, c->group, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
                             &sched, sizeof sched)) != 0) FAIL("group SCHEDULE");
  }

  /*
   * The work-submit token.
   *
   * kfifoGenerateWorkSubmitTokenHal_GA100 refuses with NV_ERR_INVALID_STATE
   * unless the channel is already assigned to a runlist, so a token coming back
   * at all is proof the channel is schedulable. The value is
   * (runlistId << 16) | chId per NV_CTRL_VF_DOORBELL_{RUNLIST_ID,VECTOR}.
   */
  {
    struct { NvU32 workSubmitToken; } tok;
    memset(&tok, 0, sizeof tok);
    if ((rc = aether_control(d, c->handle,
                             NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
                             &tok, sizeof tok)) != 0)
      FAIL("GET_WORK_SUBMIT_TOKEN");
    c->token = tok.workSubmitToken;
  }

  /*
   * USERD, the RM-allocated way.
   *
   * We are not passing hUserdMemory: RM allocates USERD itself, which is what a
   * working CUDA process does. The doorbell page still has to be reachable from
   * the host to write GP_PUT, and the only handle naming it is the CHANNEL
   * itself -- so the channel object is what gets mapped.
   *
   * CUDA never issues this map, so this is not "what the reference driver
   * does". It is the reachable route given that RM owns the buffer, and it is
   * being tried precisely because supplying our own buffer -- which RM ACCEPTS
   * with NV_OK -- produced a GP_PUT the hardware never read.
   */
  c->userd.size = HERMES_USERD_BYTES; /* RM rejects more: NV_ERR_INVALID_LIMIT */
  c->userd.handle = c->handle;
  c->userd.location = GAIA_VIDMEM;
  c->userd.hostFd = -1;
  c->userd.mapFlags = GAIA_MAP_FLAGS_VIDMEM;
  if ((rc = gaia_map_host(d, &c->userd)) != 0) FAIL("RM userd map_host");

  /* Our slot within the shared page. Must come after the token is known. */
  c->userdSlot = HERMES_USERD_SLOT(c->token);

  c->gpfifoEntries = GPFIFO_ENTRIES;
  c->push = (NvU32 *)c->pushbuffer.hostPtr;
  return 0;

fail:
  {
    /* close() zeroes the struct; the diagnosis has to outlive it. */
    const char *stage = c->failStage;
    const int status = c->failStatus;
    hermes_channel_close(d, c);
    c->failStage = stage;
    c->failStatus = status;
  }
  return rc;
}

void hermes_channel_close(aether_device *d, hermes_channel *c) {
  if (c->compute) { aether_free(d, c->compute); c->compute = 0; }
  if (c->handle) { aether_free(d, c->handle); c->handle = 0; }
  if (c->ctxshare) { aether_free(d, c->ctxshare); c->ctxshare = 0; }
  if (c->group) { aether_free(d, c->group); c->group = 0; }
  gaia_free(d, &c->usermode);
  c->doorbell = NULL;
  /* userd aliases the channel handle, which was freed above -- release only the
   * mapping, never the object. */
  c->userd.handle = 0;
  gaia_free(d, &c->userd);
  gaia_free(d, &c->errnotif);
  gaia_free(d, &c->pushbuffer);
  gaia_free(d, &c->gpfifo);
  memset(c, 0, sizeof *c);
}
