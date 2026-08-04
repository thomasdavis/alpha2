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

void hermes_begin(hermes_channel *c) {
  c->push = (NvU32 *)c->pushbuffer.hostPtr;
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
  hermes_method(c, 0, NVC56F_SEMAPHOREA, 4);
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
 * All of that is verified working: the usermode object allocates, the 64 KiB
 * doorbell page maps, BIND and GPFIFO_SCHEDULE both return NV_OK, and the token
 * comes back as 0x4. What is NOT yet working is the GPU actually consuming the
 * entry -- see the note in hermes_submit.
 */
int hermes_submit(aether_device *d, hermes_channel *c) {
  (void)d;
  NvU32 *base = (NvU32 *)c->pushbuffer.hostPtr;
  const NvU32 dwords = (NvU32)(c->push - base);
  if (dwords == 0) return 0;

  const NvU64 addr = c->pushbuffer.gpuAddr;
  NvU32 *ring = (NvU32 *)c->gpfifo.hostPtr;

  /* GP_ENTRY0_GET is bits 31:2 and holds the address directly -- the low two
   * bits are the FETCH field, and a pushbuffer is dword-aligned anyway. */
  ring[c->put * 2 + 0] = (NvU32)(addr & 0xfffffffcu);
  ring[c->put * 2 + 1] =
      (NvU32)((addr >> 32) & 0xff) | ((dwords & 0x1fffff) << 10);

  c->put = (c->put + 1) % c->gpfifoEntries;

  /* The entry must be visible before put advances, or the GPU can fetch a slot
   * we have not finished writing. */
  __asm__ __volatile__("sfence" ::: "memory");

  /*
   * NOT YET WORKING, and this is where it stops.
   *
   * Ringing the doorbell (see hermes_ring) with a valid token does not cause
   * the GPU to consume the entry. The semaphore release never lands. Everything
   * observable is correct:
   *
   *   pushbuffer  [0] 0x20040004  header: INC_METHOD, count 4, SEMAPHOREA>>2
   *               [1] 0x00000000  address hi
   *               [2] 0x00220000  address lo
   *               [3] 0xcafebabe  payload
   *               [4] 0x00000002  OPERATION_RELEASE
   *   gpfifo[0]   0x00210000      pushbuffer address
   *   gpfifo[1]   0x00001400      length 5 dwords << 10
   *   token       0x4
   *
   * Ruled out by probe: GP_PUT is not the problem -- writing put to EVERY
   * 4-byte offset across the whole 512-byte USERD, ringing the doorbell after
   * each, triggers nothing.
   *
   * The most likely remaining cause is that we are flying blind. hObjectError
   * is left zero, so no error notifier is attached, and the container has no
   * dmesg, so a channel fault produces no visible signal anywhere. The next
   * step is to attach an error notifier and read it, rather than to guess at
   * more offsets -- exactly the "read WHICH error came back" method that
   * unblocked every previous layer, applied to a path that currently returns
   * no error at all.
   */
  return 0;
}

void hermes_ring(hermes_channel *c, volatile NvU32 *userd, volatile NvU32 *doorbell,
                 NvU32 token) {
  /* USERD is 512 bytes: mapping it at 4096 returns NV_ERR_INVALID_LIMIT, at
   * 512 or below it returns NV_OK. */
  userd[HERMES_USERD_GP_PUT / 4] = c->put;
  __asm__ __volatile__("sfence" ::: "memory");
  doorbell[HERMES_DOORBELL_OFFSET / 4] = token;
  __asm__ __volatile__("sfence" ::: "memory");
}
