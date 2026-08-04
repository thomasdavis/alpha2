/*
 * channel.h — the path work travels to the GPU.
 *
 * WHAT: allocates a GPFIFO channel and the buffers it needs.
 *
 * WHY a channel is not a queue: the GPU pulls work rather than being pushed it.
 * A channel is a ring of GPFIFO entries, each of which POINTS AT a pushbuffer
 * containing methods. The host writes methods into a pushbuffer, appends an
 * entry naming that region, then advances a "put" pointer the GPU is watching.
 * Nothing is copied and no syscall is involved in submission -- which is why
 * this can be fast, and why getting the memory visibility right matters more
 * than it would for an API call.
 *
 *     pushbuffer   [ method, method, method, ... ]
 *                      ^
 *     GPFIFO       [ entry -> (gpuAddr, length) ][ ... ]
 *                      ^ put                ^ get
 *     USERD        the doorbell page: writing `put` here is the submission
 *
 * All three live in GPU-visible memory allocated through Gaia, which is why
 * Gaia had to work first.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no scheduling, no multiple channels, no
 * preemption, no error recovery. One channel, used synchronously.
 *
 * PROVENANCE: NV_CHANNEL_ALLOC_PARAMS from sdk/nvidia/inc/alloc/alloc_channel.h;
 * method encoding from class/clc56f.h (AMPERE_CHANNEL_GPFIFO_A).
 */
#ifndef HELIOS_HERMES_CHANNEL_H
#define HELIOS_HERMES_CHANNEL_H

#include "../gaia/memory.h"

/* AMPERE_USERMODE_A (clc561.h). The window it maps is NV_VIRTUAL_FUNCTION,
 * declared in swref/published/turing/tu102/dev_vm.h as 0x0003FFFF:0x00030000 —
 * 64 KiB. Ampere inherits the Turing layout. */
#define HERMES_USERMODE_CLASS 0xc561
#define HERMES_USERMODE_BYTES 65536

/* USERD is 512 bytes (mapping it larger returns NV_ERR_INVALID_LIMIT).
 *
 * It also sits at a NON-ZERO OFFSET inside its page: RM returns
 * pLinear = 0xbfef0800 on this GPU, a BAR address whose low bits are 0x800.
 * Whether mmap() hands back the page base (so USERD is at +0x800) or the object
 * itself (so it is at +0) is UNRESOLVED -- both were tried. Recorded because the
 * offset is real and any future attempt has to decide which it is.
 *
 * THE FIELD OFFSETS ARE NOT GUESSES. From swref/published/ampere/ga100/dev_ram.h,
 * where the fields are declared as bit ranges over the whole structure and the
 * word index is therefore the byte offset divided by four:
 *
 *   NV_RAMUSERD_PUT      (16*32+31):(16*32+0)   word 16 -> 0x40
 *   NV_RAMUSERD_GET      (17*32+31):(17*32+0)   word 17 -> 0x44
 *   NV_RAMUSERD_GP_GET   (34*32+31):(34*32+0)   word 34 -> 0x88
 *   NV_RAMUSERD_GP_PUT   (35*32+31):(35*32+0)   word 35 -> 0x8c
 *
 * WHY THIS COMMENT EXISTS: 0x40 was used as GP_PUT for the entire first
 * investigation. 0x40 is NV_RAMUSERD_PUT -- the LEGACY pre-GPFIFO pushbuffer
 * pointer, which a GPFIFO channel does not read. So GP_PUT stayed zero, the host
 * had no work to fetch, and GP_GET never moved. Every "the GPU does not consume
 * the entry" observation traces back to this one wrong constant.
 *
 * The lesson worth keeping: the offset came from an assumption about a struct
 * layout, and the hardware header that defines it was three greps away. */
#define HERMES_USERD_BYTES 512
#define HERMES_USERD_GP_GET 0x88
#define HERMES_USERD_GP_PUT 0x8c

/*
 * EIGHT CHANNELS SHARE ONE USERD PAGE, and the channel's own slot is not at
 * offset zero.
 *
 * alloc_channel.h says it outright while documenting a flag we never set:
 * "value 3 means the 4th channel within a USERD page. Given the USERD size is
 * 512B, we will have 8 channels total, so 3 bits". So a 4 KiB page holds eight
 * 512-byte USERD blocks, indexed by the low three bits of the channel id.
 *
 * The channel id comes out of the work-submit token, which packs it as
 * NV_CTRL_VF_DOORBELL_VECTOR (11:0) with the runlist above it. On this GPU the
 * token is 0x4, so chId 4, so our USERD starts at 4 * 512 = 0x800.
 *
 * This retro-explains an observation recorded early and left unexplained: RM
 * returned pLinear = 0xbfef0800 for the mapping, and the note at the time asked
 * whether the 0x800 meant the page base or the object. It meant neither -- it is
 * the channel's OFFSET INTO the shared page, and it is exactly 4 * 512.
 */
#define HERMES_USERD_PAGE 4096
#define HERMES_USERD_SLOT(token) ((((token) & 0xfffu) & 7u) * HERMES_USERD_BYTES)

typedef struct {
  NvHandle group;    /* the channel group (TSG) the channel lives in */
  NvHandle ctxshare; /* FERMI_CONTEXT_SHARE_A — binds the channel to an
                      * address-space context; the channel is inert without it */
  NvHandle handle;   /* the channel object */
  NvHandle compute;  /* AMPERE_COMPUTE_B, bound to this channel */
  NvHandle copy;     /* AMPERE_DMA_COPY_B — a working driver allocates one on
                      * every compute channel */

  gaia_buffer usermode;   /* AMPERE_USERMODE_A — not memory but a 64 KiB MMIO
                           * window over NV_VIRTUAL_FUNCTION, which maps through
                           * the same path as a memory object */
  gaia_buffer gpfifo;     /* the ring of entries */
  gaia_buffer pushbuffer; /* the methods those entries point at */
  gaia_buffer errnotif;   /* hObjectError — RM writes fault records here */
  gaia_buffer userd;      /* our own USERD, so GP_PUT's location is certain */

  volatile NvU32 *doorbell; /* the mapped usermode window */
  NvU32 token;              /* this channel's work-submit token */
  NvU32 userdSlot;          /* byte offset of OUR USERD within the shared page */

  NvU32 gpfifoEntries;
  NvU32 put;   /* our index into the ring */
  NvU32 *push; /* write cursor into the pushbuffer */

  /* Which step of bring-up failed, for the same reason aether_device carries
   * one: open() has fifteen ways to fail and one return value. */
  const char *failStage;
  int failStatus;
} hermes_channel;

/*
 * The error notifier.
 *
 * hObjectError on BOTH the channel group and the channel takes a memory object
 * handle, and RM writes NvNotification records into it. Attaching one is not
 * optional for development: without it a channel fault produces no signal at
 * all, and inside a container there is no dmesg either, so every call returns
 * NV_OK and nothing happens.
 *
 * It stays because a fault here is otherwise completely silent: RM returns
 * NV_OK, the container has no dmesg, and the channel simply does nothing.
 *
 * NOTE also: NVA06F_CTRL_CMD_SET_ERROR_NOTIFIER takes NVA06F_CTRL_SET_ERROR_NOTIFIER_PARAMS
 * { NvBool bNotifyEachChannelInTSG } -- a single byte. Passing an NvU32 gives
 * NV_ERR_INVALID_ARGUMENT. Attaching via hObjectError at allocation time is
 * what actually works and is sufficient.
 *
 * On USERD: RM allocates it, and the channel object is the handle that names
 * it. RM will ALSO accept a client-supplied buffer through hUserdMemory[0] --
 * with NV_OK -- but the hardware then still reads RM's own, so GP_PUT written
 * into the client buffer is never seen and the channel never fetches. That
 * acceptance is the single most expensive false signal in this whole exercise:
 * it made a wrong design look confirmed. A working CUDA process leaves
 * hUserdMemory zero, and so do we.
 */

/*
 * Allocate a channel and everything it needs, and leave it ready to submit:
 * group, context share, channel, compute object, error notifier, our own USERD,
 * the usermode doorbell window, BIND, SCHEDULE, and the work-submit token.
 *
 * All of that is one function because none of it is independently useful. A
 * channel that is allocated but not scheduled, or scheduled but has no token,
 * is not a channel you can hand work to -- and splitting the steps was how the
 * first version ended up with a half-configured channel that returned NV_OK
 * from every call it made.
 */
int hermes_channel_open(aether_device *d, hermes_channel *c);

void hermes_channel_close(aether_device *d, hermes_channel *c);

#endif /* HELIOS_HERMES_CHANNEL_H */
