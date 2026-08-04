/*
 * pushbuffer.c — see pushbuffer.h.
 */
#include "pushbuffer.h"
#include "../aether/ioctl.h"

#include <string.h>

/* clc56f.h method encoding */
#define SEC_OP_INC_METHOD 1
#define METHOD_HDR(op, count, sub, addr)                                       \
  (((NvU32)(op) << 29) | ((NvU32)(count) << 16) | ((NvU32)(sub) << 13) |       \
   ((NvU32)(addr) >> 2))

/* clc56f.h host methods */
#define NVC56F_SEMAPHOREA 0x00000010
#define NVC56F_SEMAPHORED_OPERATION_RELEASE 0x00000002
/* RELEASE_WFI_EN is 0 and means "wait for idle before releasing", which is
 * exactly what a completion signal wants: the value must not land before the
 * work it is reporting on. */
#define SEM_RELEASE_WFI_EN (0u << 20)

/*
 * Each segment gets FRESH pushbuffer space; it does not reuse the start.
 *
 * The first version reset the cursor to the base every time, which silently
 * overwrites methods the GPU may still be fetching -- submitting a second
 * kernel while the first is in flight rewrites the memory the PBDMA is reading
 * from. It manifested as ROBUST_CHANNEL_PBDMA_ERROR (32) on the second launch,
 * after which the channel is dead and every later kernel reports whatever the
 * notifier still holds. The first kernel always passed, which made it look like
 * a problem with kernels 2 through 5 rather than with the ring underneath them.
 *
 * Bump, aligned, and wrap when there is not enough room left. Wrapping is only
 * safe because callers wait for each kernel's effect before submitting the
 * next; a pipelined submitter needs real tracking of what the GPU has consumed.
 */
#define HERMES_SEGMENT_ALIGN 64 /* dwords */

void hermes_begin(hermes_channel *c) {
  NvU32 *base = (NvU32 *)c->pushbuffer.hostPtr;
  const NvU32 capacity = (NvU32)(c->pushbuffer.size / 4);
  if (c->pushOffset + HERMES_SEGMENT_ALIGN * 4 > capacity) c->pushOffset = 0;
  c->push = base + c->pushOffset;
}

void hermes_method(hermes_channel *c, NvU32 subchannel, NvU32 addr, NvU32 count) {
  *c->push++ = METHOD_HDR(SEC_OP_INC_METHOD, count, subchannel, addr);
}

void hermes_data(hermes_channel *c, NvU32 value) { *c->push++ = value; }

void hermes_semaphore_release(hermes_channel *c, NvU64 gpuAddr, NvU32 payload) {
  /* Four consecutive methods starting at SEMAPHOREA, so one INC_METHOD header
   * with count 4 covers all of them:
   *   SEMAPHOREA  address bits 63:32
   *   SEMAPHOREB  address bits 31:0
   *   SEMAPHOREC  payload
   *   SEMAPHORED  operation */
  /*
   * Subchannel 1, not 0.
   *
   * SEMAPHOREA..D are host methods, but the PBDMA still rejects a method stream
   * that names a subchannel with no object bound to it: once SET_OBJECT has
   * put the compute class on subchannel 1, emitting on subchannel 0 raises
   * ROBUST_CHANNEL_PBDMA_ERROR (32). It worked before compute existed precisely
   * because nothing was bound anywhere, which made subchannel 0 as good as any.
   */
  hermes_method(c, 1, NVC56F_SEMAPHOREA, 4);
  hermes_data(c, (NvU32)(gpuAddr >> 32));
  hermes_data(c, (NvU32)(gpuAddr & 0xffffffffu));
  hermes_data(c, payload);
  hermes_data(c, NVC56F_SEMAPHORED_OPERATION_RELEASE | SEM_RELEASE_WFI_EN);
}

/*
 * Submission on Volta and later is NOT just "advance GP_PUT".
 *
 * The pre-Volta model was: write the ring, bump GP_PUT in USERD, and the host
 * engine notices. Volta introduced work-submission tokens: each channel has a
 * token, obtained with NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, and
 * submission means writing that token to a doorbell register inside a USERMODE
 * object (AMPERE_USERMODE_A, 0xc561) at NOTIFY_CHANNEL_PENDING (0x90).
 *
 * The doorbell offset is not a guess either. The usermode object maps the
 * register window NV_VIRTUAL_FUNCTION, which swref/published/turing/tu102/dev_vm.h
 * declares as 0x0003FFFF:0x00030000 -- 64 KiB based at 0x30000. Within it:
 *
 *   NV_VIRTUAL_FUNCTION_TIME_0     0x30080   -> +0x80
 *   NV_VIRTUAL_FUNCTION_DOORBELL   0x30090   -> +0x90
 *
 * Reading a ticking GPU clock at +0x80 is therefore an independent confirmation
 * that the window is mapped where we think it is, which is why that probe was
 * worth running before trusting +0x90.
 *
 * The token's own layout, from dev_ctrl.h and kfifoGenerateWorkSubmitTokenHal_GA100:
 *
 *   NV_CTRL_VF_DOORBELL_VECTOR       11:0    the channel id
 *   NV_CTRL_VF_DOORBELL_RUNLIST_ID  22:16    the runlist the channel is on
 *
 * so token 0x4 means channel 4 on runlist 0. RM refuses to generate a token at
 * all unless the channel is already assigned to a runlist, which makes a
 * successful GET_WORK_SUBMIT_TOKEN proof that the channel is schedulable.
 */
int hermes_submit(aether_device *d, hermes_channel *c) {
  (void)d;
  NvU32 *base = (NvU32 *)c->pushbuffer.hostPtr + c->pushOffset;
  const NvU32 dwords = (NvU32)(c->push - base);
  if (dwords == 0) return 0;

  const NvU64 addr = c->pushbuffer.gpuAddr + (NvU64)c->pushOffset * 4;
  NvU32 *ring = (NvU32 *)c->gpfifo.hostPtr;

  /* GP_ENTRY0_GET is bits 31:2 and holds the address directly -- the low two
   * bits are the FETCH field, and a pushbuffer is dword-aligned anyway. */
  ring[c->put * 2 + 0] = (NvU32)(addr & 0xfffffffcu);
  ring[c->put * 2 + 1] =
      (NvU32)((addr >> 32) & 0xff) | ((dwords & 0x1fffff) << 10);

  c->put = (c->put + 1) % c->gpfifoEntries;
  /* Advance past this segment, aligned, so the next one cannot land on it. */
  c->pushOffset += (dwords + HERMES_SEGMENT_ALIGN - 1) & ~(HERMES_SEGMENT_ALIGN - 1);

  /* The entry must be visible before put advances, or the GPU can fetch a slot
   * we have not finished writing. */
  __asm__ __volatile__("sfence" ::: "memory");

  /* Submission itself is hermes_ring: GP_PUT, then the doorbell. Kept separate
   * so a caller can build several entries and ring once. */
  return 0;
}

void hermes_ring(hermes_channel *c, volatile NvU32 *userd, volatile NvU32 *doorbell,
                 NvU32 token) {
  /* `userd` is the base of the shared 4 KiB page; our channel's 512-byte block
   * starts at userdSlot. Writing at the page base addresses whichever channel
   * happens to occupy slot 0 -- which is not an error, just someone else's
   * doorbell. */
  userd += c->userdSlot / 4;
  userd[HERMES_USERD_GP_PUT / 4] = c->put;
  __asm__ __volatile__("sfence" ::: "memory");
  doorbell[HERMES_DOORBELL_OFFSET / 4] = token;
  __asm__ __volatile__("sfence" ::: "memory");
}
