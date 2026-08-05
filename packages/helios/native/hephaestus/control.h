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
#include <stdlib.h>

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
 * The safe default: stall long enough for a dependent consumer, set no
 * barriers, wait on nothing.
 *
 * This was 15, the maximum, on the reasoning that an unscheduled kernel that is
 * slow beats one that races. The reasoning stands; the CONSTANT was never
 * measured, and it cost 30x. A step's 184 kernels spent ~32 ms on the GPU where
 * the same count of a 1024-element add spends ~4, and the difference was issue
 * latency rather than arithmetic -- fifteen cycles of nothing between every
 * pair of instructions, in kernels too small to hide it behind other warps.
 *
 * tools/stall_probe.c asks the hardware instead. It builds a chain where each
 * instruction reads what the one before it wrote, sweeps the stall down, and
 * finds where the answer stops being right:
 *
 *     IADD3 / IMAD / FFMA / SHF+LOP3     4
 *     MOV c[]                            5
 *     IMAD.WIDE, HADD2                   0
 *     ISETP -> @P                       13   <- see sm86_flow.c
 *
 * 7 is the worst of what this default actually governs (5) plus margin. ISETP
 * is NOT governed by it: the gap is too large to cover with one number, so that
 * emitter clamps its own stall and this default cannot lower it.
 *
 * WHY A SINGLE WARP IS THE WORST CASE, and therefore why a probe is enough:
 * the stall spaces one warp's own instruction stream. Additional resident warps
 * interleave their instructions between the pair and can only ADD delay. The
 * probe runs one block of one thread, which is the least slack the hardware
 * will ever offer.
 *
 * This is still not a scheduler. A scheduler would know which pairs are
 * independent and stall 1 between them; this stalls as if every pair were
 * dependent, because nothing here tracks registers. That remains on the table
 * and is worth roughly another 2x on ALU-bound kernels.
 */
/*
 * MEASURED, 2026-08-05: this stack is NOT stall-bound, so the scheduler on the
 * table above would not pay for itself.
 *
 * Sweeping the default over a whole training step at batch 128 moves nothing:
 *
 *     stall 7   212.3 ms/step   19,294 tok/s
 *     stall 6   214.5 ms/step   19,096 tok/s
 *     stall 5   215.3 ms/step   19,027 tok/s
 *
 * -- inside the run-to-run spread, and monotonically the wrong way. The note
 * above estimates a real scheduler at "roughly another 2x on ALU-bound
 * kernels"; whatever this step is bound by, it is not instruction issue, and
 * these kernels are not ALU-bound at this size.
 *
 * Stall 5 also moved the LOSS, 4.1903 to 4.1893, which is the more useful half
 * of the result: the margin in 7 is doing something. Lowering it is not a free
 * setting even where it looks like one.
 */
static inline hp_control hp_ctrl_safe(void) {
  hp_control c = {7, 0, HP_NO_BARRIER, HP_NO_BARRIER, 0, 0};
  return c;
}

/* Variable-latency instructions (loads, S2R, MUFU) must set a write barrier and
 * their consumer must wait on it. This pairs with hp_ctrl_wait(). */
static inline hp_control hp_ctrl_setbar(unsigned barrier) {
  hp_control c = {1, 0, barrier, HP_NO_BARRIER, 0, 0};
  return c;
}

/* Wait for the given barrier before issuing.
 *
 * The wait covers this instruction's INPUT; the stall covers its output, for
 * whoever reads it next -- two different hazards on one instruction. So the
 * stall here is the same measured ALU figure as hp_ctrl_safe and moves with it,
 * while the wait mask does the job the barrier was set for. */
static inline hp_control hp_ctrl_wait(unsigned barrier) {
  hp_control c = {7, 0, HP_NO_BARRIER, HP_NO_BARRIER, 1u << barrier, 0};
  return c;
}

/*
 * Set a READ barrier: signal when this instruction has finished READING its
 * source registers, so a later instruction may safely overwrite them.
 *
 * The other direction of the same missing interlock. hp_ctrl_setbar covers
 * write-after-read on a variable-latency RESULT -- do not read R before the load
 * writing it lands. This covers write-after-read on a variable-latency
 * instruction's OPERANDS -- do not overwrite R before the store issuing it has
 * read it. A global store holds its address and data registers until the memory
 * pipe accepts them, which is not when it issued.
 *
 * WHY NOTHING NEEDED IT UNTIL NOW: every store in this stack was the last
 * instruction before EXIT, so no later instruction could overwrite anything. The
 * first kernel to store in the MIDDLE -- layerNorm's backward, which writes xhat
 * and then keeps going -- reused the address register pair for its second store
 * and corrupted the first. It was correct at 64 blocks and wrong at 256, because
 * whether the pipe is still holding the register depends on how much traffic is
 * queued ahead of it, and that is exactly the kind of load-dependent wrongness
 * this whole file exists to prevent.
 */
static inline hp_control hp_ctrl_setread(unsigned barrier) {
  hp_control c = {1, 0, HP_NO_BARRIER, barrier, 0, 0};
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

/*
 * The control field a BRANCH wants: stall 5, yield set.
 *
 * Every other instruction in this stack uses stall 15 with yield clear, which
 * is maximally conservative and works. A branch is the exception, and it is not
 * a performance preference -- a backward BRA carrying stall 15 / yield 0 faults
 * the channel, both in a matmul and in a four-instruction loop that does
 * nothing but add a register to itself. ptxas emits stall 5 with yield set for
 * every BRA it generates, so that is what this returns.
 *
 * The values are copied from ptxas rather than reasoned about. What the yield
 * bit means for a branch specifically is not something this file can claim to
 * know; what it can say is that the combination ptxas uses works and the
 * conservative-looking one does not.
 */
static inline hp_control hp_ctrl_branch(void) {
  hp_control c = {5, 1, HP_NO_BARRIER, HP_NO_BARRIER, 0, 0};
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
