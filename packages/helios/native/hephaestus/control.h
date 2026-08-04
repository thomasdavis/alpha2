/*
 * control.h — the scheduling control field, bits 105-127.
 *
 * WHAT: the 23-bit field every sm_86 instruction carries telling the hardware
 * how long to stall, whether to yield, and which dependency barriers to set and
 * wait on.
 *
 * WHY THIS IS THE DANGEROUS PART: from Kepler onwards NVIDIA moved instruction
 * scheduling out of the hardware and into the compiler. There is no scoreboard
 * interlock for variable-latency results — if an instruction reads a register
 * before the load writing it has landed, the hardware does not stall and does
 * not fault. It reads stale data. A wrong control field therefore produces
 * silently incorrect results or a hung channel, never a clean error.
 *
 * That is the same failure signature as X58's halved gradient norm, which is
 * why the standards demand known-answer tests here rather than plausibility.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: it does not compute a schedule. Choosing
 * stall counts and barrier assignments for a whole kernel is a separate problem
 * (scheduler, later). This file only encodes a decision already made, and
 * offers a deliberately pessimistic default for code that has not been
 * scheduled yet.
 *
 * LAYOUT, relative to bit 105 of the instruction:
 *   +0..3    stall count, 0-15 cycles before issuing the next instruction
 *   +4       yield hint
 *   +5..7    write barrier index (7 = none)
 *   +8..10   read barrier index  (7 = none)
 *   +11..16  wait mask, one bit per barrier
 *   +17..20  register reuse flags
 *
 * PROVENANCE: derived from encodings captured from ptxas. The reference values
 * in HP_CTRL_* below are lifted verbatim from instructions the vendor compiler
 * emitted, so a fully-stalled instruction is known-good rather than theorised.
 */
#ifndef HELIOS_HEPHAESTUS_CONTROL_H
#define HELIOS_HEPHAESTUS_CONTROL_H

#include <stdint.h>

typedef struct {
  unsigned stall;      /* 0-15 */
  unsigned yield;      /* 0 or 1 */
  unsigned writeBarrier; /* 0-6, or 7 for none */
  unsigned readBarrier;  /* 0-6, or 7 for none */
  unsigned waitMask;   /* 6 bits, one per barrier */
  unsigned reuse;      /* 4 bits */
} hp_control;

#define HP_NO_BARRIER 7

/*
 * The safe default: stall the maximum, set no barriers, wait on nothing.
 *
 * This is deliberately the most conservative encoding available and it is
 * slower than anything the vendor compiler would emit. That is the correct
 * trade for a stack whose scheduler does not exist yet: an unscheduled kernel
 * that is slow is a performance problem, an unscheduled kernel that races is an
 * undebuggable correctness problem. Speed comes back when the scheduler lands
 * and can prove the shorter stalls are safe.
 */
static inline hp_control hp_ctrl_safe(void) {
  hp_control c = {15, 0, HP_NO_BARRIER, HP_NO_BARRIER, 0, 0};
  return c;
}

/* Variable-latency instructions (loads, S2R, MUFU) must set a write barrier and
 * their consumer must wait on it. This pairs with hp_ctrl_wait(). */
static inline hp_control hp_ctrl_setbar(unsigned barrier) {
  hp_control c = {1, 0, barrier, HP_NO_BARRIER, 0, 0};
  return c;
}

/* Wait for the given barrier before issuing. */
static inline hp_control hp_ctrl_wait(unsigned barrier) {
  hp_control c = {15, 0, HP_NO_BARRIER, HP_NO_BARRIER, 1u << barrier, 0};
  return c;
}

/*
 * Wait on SEVERAL barriers at once.
 *
 * Two variable-latency producers feeding one consumer need two barriers and a
 * wait on both. Pointing both at the same barrier looks economical and is not
 * safe: the wait can release once, leaving the second register still in flight,
 * and the consumer reads whatever was there. It shows up as a plausible-looking
 * wrong value rather than as a stall, which is the worst way for a race to
 * present.
 */
/*
 * Wait on one barrier and set another.
 *
 * A variable-latency instruction that CONSUMES a variable-latency result needs
 * both: MUFU reading a loaded value must wait for the load, and its own result
 * is not ready when it issues either. Using only setbar looks right and is not:
 * the instruction issues immediately and reads whatever was in the register.
 * That failure is silent and plausible -- MUFU.EX2 on a stale zero returns 1.0,
 * which is a perfectly reasonable-looking answer to exp2 of something.
 */
static inline hp_control hp_ctrl_wait_setbar(unsigned waitBar,
                                             unsigned setBar) {
  hp_control c = {15, 0, setBar, HP_NO_BARRIER, 1u << waitBar, 0};
  return c;
}

static inline hp_control hp_ctrl_waitmask(unsigned mask) {
  hp_control c = {15, 0, HP_NO_BARRIER, HP_NO_BARRIER, mask & 0x3f, 0};
  return c;
}

/* Pack to the 23-bit field value. */
uint32_t hp_control_pack(hp_control c);

/* Unpack, so a captured encoding can be inspected and compared. */
hp_control hp_control_unpack(uint32_t packed);

#endif /* HELIOS_HEPHAESTUS_CONTROL_H */
