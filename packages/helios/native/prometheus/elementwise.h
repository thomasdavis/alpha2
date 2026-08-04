/*
 * elementwise.h — one generator for the whole elementwise family.
 *
 * WHAT: emits `out[i] = f(in[i], i)` for a choice of f, where i is the global
 * thread index.
 *
 * WHY one generator: the existing TypeScript stack has 46 elementwise kernel
 * generators, and they are the same eleven instructions with one changed in the
 * middle — compute a global index, turn it into two addresses, load, apply an
 * operation, store. Most of the 46 are also vectorisation variants (Vec4,
 * Vec4x2) of the same arithmetic rather than different arithmetic. So the
 * skeleton is written once and the operation is a parameter, which is what
 * makes porting the family a list of cases rather than a list of programs.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no bounds check. Every kernel here is
 * launched with exactly as many threads as there are elements, so a guard would
 * be dead code that still has to be right. Sizes that are not a multiple of the
 * block need predication, and that is a separate piece of work with its own
 * tests rather than something to smuggle in now.
 *
 * REGISTERS are assigned by hand and deliberately wastefully:
 *   R0    global index          R5    element size in bytes
 *   R2:R3 input address         R6:R7 output address
 *   R4    loaded value          R8    result
 * A register allocator is its own problem; coupling it to this one would make
 * both harder to get right.
 */
#ifndef HELIOS_PROMETHEUS_ELEMENTWISE_H
#define HELIOS_PROMETHEUS_ELEMENTWISE_H

#include "kernel.h"

/*
 * The operation applied to each element.
 *
 * Named for what the HARDWARE does, not for what a math library calls it:
 * MUFU computes exp2 and log2, so those are the names. Calling them exp and log
 * would imply a base conversion this generator does not perform, and a kernel
 * whose name lies about its semantics is worse than one with an awkward name.
 */
typedef enum {
  PR_EW_COPY,      /* out[i] = in[i]                    */
  PR_EW_ADD_INDEX, /* out[i] = in[i] + i                */
  PR_EW_ADD_CONST, /* out[i] = in[i] + 0x1234           */
  PR_EW_FADD,      /* out[i] = in[i] + in[i]            */
  PR_EW_FMUL,      /* out[i] = in[i] * in[i]            */
  PR_EW_FFMA,      /* out[i] = in[i]*in[i] + in[i]      */
  PR_EW_FNEG,      /* out[i] = -in[i]                   */
  PR_EW_EXP2,      /* out[i] = exp2(in[i])              */
  PR_EW_LOG2,      /* out[i] = log2(in[i])              */
  PR_EW_RCP,       /* out[i] = 1 / in[i]                */
  PR_EW_RSQ,       /* out[i] = 1 / sqrt(in[i])          */
  PR_EW_RELU,      /* out[i] = max(in[i], 0)            */
  PR_EW_INDEX,     /* out[i] = i — a probe, reads nothing */

  /* Binary: out[i] = a[i] OP b[i]. */
  PR_EW_ADD,       /* a + b                              */
  PR_EW_SUB,       /* a - b                              */
  PR_EW_MUL,       /* a * b                              */
  PR_EW_DIV,       /* a * (1/b), which is what the hardware offers */

  /* Scalar: out[i] = a[i] OP s, with s from the constant bank. */
  PR_EW_SCALE,     /* a * s                              */

  /*
   * Composed unary. The hardware has exp2 and log2, not exp and log, so these
   * are two or three instructions rather than one -- a base conversion around a
   * MUFU. Naming them exp and log is honest here because that IS what they
   * compute; the earlier PR_EW_EXP2 stays because it names a different thing.
   */
  PR_EW_EXP,       /* exp2(a * log2(e))                  */
  PR_EW_LOG,       /* log2(a) * ln(2)                    */
  PR_EW_SQRT,      /* 1 / rsqrt(a)                       */

  PR_EW_COUNT,
} pr_ew_op;

/* Emit the kernel into `prog`, returning the instruction count. */
unsigned pr_emit_elementwise(hp_word *prog, pr_ew_op op);

#endif /* HELIOS_PROMETHEUS_ELEMENTWISE_H */
