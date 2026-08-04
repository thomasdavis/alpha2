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

typedef struct {
  NvHandle group;    /* the channel group (TSG) the channel lives in */
  NvHandle handle;   /* the channel object */
  NvHandle compute;  /* AMPERE_COMPUTE_B, bound to this channel */

  gaia_buffer gpfifo;     /* the ring of entries */
  gaia_buffer pushbuffer; /* the methods those entries point at */
  /* No userd buffer: RM allocates the doorbell page itself, and supplying one
   * makes channel allocation fail. */

  NvU32 gpfifoEntries;
  NvU32 put;   /* our index into the ring */
  NvU32 *push; /* write cursor into the pushbuffer */
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
 * Observed on an RTX 3070 after a submission that did not execute -- record 1
 * of the notifier buffer:
 *
 *   18c89c9f da551460 00000004 ffff0000
 *   \_______ ______/  \__ __/  \__ __/
 *           v            v        v
 *      timestamp      info32    status
 *
 * The timestamp is real, which is the useful part: it proves the GPU side
 * wrote into memory we allocated, so the notifier plumbing is correct and the
 * hardware is reachable.
 *
 * Do NOT read more into info32 than that. nverror.h defines no robust-channel
 * error 4, so this is not an RC code, and interpreting it as one would be
 * inventing a diagnosis.
 *
 * NOTE also: NVA06F_CTRL_CMD_SET_ERROR_NOTIFIER takes NVA06F_CTRL_SET_ERROR_NOTIFIER_PARAMS
 * { NvBool bNotifyEachChannelInTSG } -- a single byte. Passing an NvU32 gives
 * NV_ERR_INVALID_ARGUMENT. Attaching via hObjectError at allocation time is
 * what actually works and is sufficient.
 *
 * On USERD: RM will allocate the doorbell page itself, and it will ALSO accept
 * ours through hUserdMemory[0] -- but only if hVASpace is left zero. The first
 * sweep tested {userd, vaspace} pairs and stopped at the first success
 * ({none, none}), never reaching {ours, none}, which also works. Owning USERD
 * is worth having because it makes GP_PUT's location certain rather than
 * assumed.
 *
 * NEITHER causes the GPU to execute submitted work. See
 * donto-resources/research/alpha-helios-reimagined/HERMES-SUBMISSION-BLOCKER.md
 * for the full map of what is established and what remains.
 */

/* Allocate a channel and everything it needs. */
int hermes_channel_open(aether_device *d, hermes_channel *c);

void hermes_channel_close(aether_device *d, hermes_channel *c);

#endif /* HELIOS_HERMES_CHANNEL_H */
