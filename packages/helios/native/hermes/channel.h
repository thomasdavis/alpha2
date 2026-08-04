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

/* Allocate a channel and everything it needs. */
int hermes_channel_open(aether_device *d, hermes_channel *c);

void hermes_channel_close(aether_device *d, hermes_channel *c);

#endif /* HELIOS_HERMES_CHANNEL_H */
