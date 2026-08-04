/*
 * pushbuffer.h — writing methods, and ringing the doorbell.
 *
 * WHAT: encodes methods into the pushbuffer, appends a GPFIFO entry pointing at
 * them, and advances the put pointer so the GPU fetches the work.
 *
 * WHY submission has no syscall in it: the GPU is watching a memory location.
 * Once the channel exists, handing it work is three memory writes -- methods,
 * an entry, then the put pointer. Nothing traps to the kernel. This is the
 * whole reason a from-scratch stack can be fast, and the reason correctness
 * here depends on memory ordering rather than on an API contract.
 *
 * The method encoding, from class/clc56f.h:
 *
 *   header = (opcode << 29) | (count << 16) | (subchannel << 13) | (addr >> 2)
 *
 * with opcode 1 = INC_METHOD, meaning consecutive data words go to consecutive
 * method addresses. `addr >> 2` because method addresses are byte offsets into
 * the class and the field holds them as dwords.
 *
 * The GPFIFO entry is two words:
 *   entry0 = pushbuffer address, bits 31:2   (GET)
 *   entry1 = address >> 32 in bits 7:0       (GET_HI)
 *          | length in DWORDS in bits 30:10  (LENGTH)
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no ring-wrap handling, no fencing beyond
 * a compiler barrier, no waiting. One submission at a time, checked by polling
 * the semaphore it releases.
 */
#ifndef HELIOS_HERMES_PUSHBUFFER_H
#define HELIOS_HERMES_PUSHBUFFER_H

#include "channel.h"

/* Start a fresh pushbuffer segment. */
void hermes_begin(hermes_channel *c);

/* Emit a method header plus `count` data words, written by the caller through
 * hermes_data(). Subchannel 0 is the compute engine. */
void hermes_method(hermes_channel *c, NvU32 subchannel, NvU32 addr, NvU32 count);

/* Append one data word. */
void hermes_data(hermes_channel *c, NvU32 value);

/*
 * Ask the GPU to write `payload` to `gpuAddr` once it reaches this point.
 *
 * This is the cheapest possible proof that the GPU executed our methods: a
 * value we chose appearing at an address we chose, written by the hardware
 * rather than by us.
 */
void hermes_semaphore_release(hermes_channel *c, NvU64 gpuAddr, NvU32 payload);

/* Close the segment, append the GPFIFO entry, and advance put. */
int hermes_submit(aether_device *d, hermes_channel *c);

/* USERD is 512 bytes (mapping it larger returns NV_ERR_INVALID_LIMIT). */
#define HERMES_USERD_BYTES 512
#define HERMES_USERD_GP_PUT 0x40

/* AMPERE_USERMODE_A (class 0xc561) is a 64 KiB register page; the channel
 * doorbell is NVC361_NOTIFY_CHANNEL_PENDING within it. */
#define HERMES_USERMODE_CLASS 0xc561
#define HERMES_USERMODE_BYTES 65536
#define HERMES_DOORBELL_OFFSET 0x90

/* Volta+ submission: bump GP_PUT, then write the channel's work-submit token to
 * the doorbell. The token comes from NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN
 * (0xc36f0108).
 *
 * NOTE: this sequence is implemented and every component of it verifies, but
 * the GPU does not yet consume the submitted entry. See hermes_submit. */
void hermes_ring(hermes_channel *c, volatile NvU32 *userd, volatile NvU32 *doorbell,
                 NvU32 token);

#endif /* HELIOS_HERMES_PUSHBUFFER_H */
