/*
 * schedule.c — making a channel runnable, and learning how to ring it.
 *
 * WHAT: the control calls that turn an allocated channel into one the hardware
 * will schedule, plus the work-submit token that names it at the doorbell.
 *
 * WHY separate from channel.c: allocating objects and scheduling them fail
 * differently. A channel that allocates but is never scheduled returns NV_OK
 * from every call and then silently does nothing, which is a distinct enough
 * problem to be worth its own file.
 *
 * WHAT A WORKING DRIVER DOES, which is less than it looks: exactly one schedule
 * call, on the GROUP. A CUDA trace of nine channels issues
 * NVA06C_CTRL_CMD_GPFIFO_SCHEDULE once and never calls the channel's own
 * schedule or either BIND. We had been issuing all four, and each returned
 * NV_OK -- which is precisely why they survived, since a redundant call that
 * succeeds looks like diligence.
 */
#include "channel.h"
#include "../aether/ioctl.h"

#include <string.h>

/* ctrla06c.h, ctrlc36f.h, ctrl2080gr.h */
#define NVA06C_CTRL_CMD_GPFIFO_SCHEDULE 0xa06c0101
#define NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN 0xc36f0108
#define NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE 0x20801210

int hermes_channel_schedule(aether_device *d, hermes_channel *c) {
  int rc;
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
                             &sched, sizeof sched)) != 0) return rc;
  }

  /*
   * Compute-instruction-level preemption mode.
   *
   * Issued on the SUBDEVICE, naming the channel GROUP -- not the channel. A
   * CUDA process calls this exactly once per group with
   *   flags = CILP_SET(1), hChannel = <the TSG>, gfxp = 0, cilp = 2
   * and nothing else in the GR control range changes state.
   *
   * PROVENANCE: read out of an ioctl trace; the params struct is
   * NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_PARAMS, 32 bytes including a
   * 16-byte grRouteInfo the trace leaves zero.
   */
  {
    struct {
      NvU32 flags;
      NvHandle hChannel;
      NvU32 gfxpPreemptMode;
      NvU32 cilpPreemptMode;
      NvU32 routeFlags;
      NvU32 pad;
      NvU64 route __attribute__((aligned(8)));
    } pm;
    memset(&pm, 0, sizeof pm);
    pm.flags = 1;              /* FLAGS_CILP_SET */
    pm.hChannel = c->group;    /* the TSG, not the channel */
    pm.cilpPreemptMode = 2;    /* as observed */
    if ((rc = aether_control(d, d->subdevice,
                             NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE, &pm,
                             sizeof pm)) != 0)
      return rc;
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
      return rc;
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
  if ((rc = gaia_map_host(d, &c->userd)) != 0) return rc;

  /* Our slot within the shared page. Must come after the token is known. */
  c->userdSlot = HERMES_USERD_SLOT(c->token);

  return 0;
}
