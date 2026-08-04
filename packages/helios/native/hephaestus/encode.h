/*
 * encode.h — the shared skeleton of a 128-bit sm_86 instruction.
 *
 * WHY IT EXISTS: the encoders are split by instruction class -- integer, float,
 * memory, control flow -- because each class has its own field layout and its
 * own provenance notes, and one file holding all of them was 312 lines of
 * unrelated bit positions. What they DO share is the opening move: opcode in
 * the low twelve bits, the always-true predicate, and the scheduling control
 * field at the top. That is this file, and it is private to hephaestus.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it encodes no operands. An operand field
 * that were common to every class would be a coincidence, not a rule.
 */
#ifndef HEPHAESTUS_ENCODE_H
#define HEPHAESTUS_ENCODE_H

#include "sm86.h"

/* Opcode, predicate PT, and the control field. Every encoder starts here. */
hp_word hp_base(unsigned opcode, hp_control c);

#endif /* HEPHAESTUS_ENCODE_H */
