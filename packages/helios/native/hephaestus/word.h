/*
 * word.h — the sm_86 instruction word.
 *
 * WHAT: a 128-bit instruction and the primitives for placing fields into it.
 *
 * WHY: every SASS instruction on Volta and later is exactly 128 bits, laid out
 * as bit-fields with no alignment or endianness surprises once you have the
 * positions. Getting a field position wrong does not fail to compile and does
 * not trap at runtime — it silently executes a different instruction. So field
 * placement is centralised here, given names, and tested against encodings
 * captured from the vendor compiler.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no knowledge of any specific instruction.
 * This file knows that a destination register lives at bit 16; it does not know
 * that MOV has one. Instruction shapes live in sm86.c.
 *
 * PROVENANCE: NVIDIA does not document SASS. Every position below was derived
 * from encodings emitted by ptxas for sm_86 and confirmed across multiple
 * samples — see scripts/sass-catalogue.mjs (captures) and
 * scripts/sass-derive.mjs (differential derivation). The derivation output is
 * checked into isa/sm86-derived-fields.json.
 *
 * THE LAYOUT, and the evidence for each field:
 *
 *   bits 0-11    opcode        EXIT=0x94d MOV=0xa02 S2R=0x919 IADD3=0x810
 *                              STG=0x986 LDG=0x981 BAR=0xb1d NOP=0x918
 *   bits 12-14   predicate     "EXIT" encodes 0x7 (PT); "@P0 EXIT" encodes 0x0
 *   bit  15      predicate NOT
 *   bits 16-23   destination   S2R R0 vs R5 moves only bits 16-18
 *   bits 24-31   source A      STG.E [R2.64] places 2 here
 *   bits 32-63   source B      register, 32-bit immediate, or const-bank ref
 *   bits 64-71   source C      IADD3's trailing RZ encodes 0xff here
 *   bits 72-79   special reg   SR_TID.X=0x21, SR_CTAID.X=0x25
 *   bits 105-127 control       scheduling; see control.h
 *
 * Const-bank operands are the reason kernel parameters work at all: they encode
 * as (byte_offset << 6) placed at bit 32, with the bank index above. MOV R1,
 * c[0x0][0x28] carries 0x28 << 6 == 0xa00 in that field.
 */
#ifndef HELIOS_HEPHAESTUS_WORD_H
#define HELIOS_HEPHAESTUS_WORD_H

#include <stdint.h>

/* nvdisasm prints instructions as two 64-bit words, low first. Keeping the same
 * split means captured encodings can be compared field-for-field against what we
 * produce without any reshuffling. */
typedef struct {
  uint64_t lo;
  uint64_t hi;
} hp_word;

/* Field positions. Named so a wrong constant is visible at the call site. */
#define HP_F_OPCODE 0
#define HP_F_PRED 12
#define HP_F_PRED_NOT 15
#define HP_F_DST 16
#define HP_F_SRCA 24
#define HP_F_SRCB 32
#define HP_F_SRCC 64
#define HP_F_SREG 72
#define HP_F_CONTROL 105

/* Register 255 is RZ, the zero register: reads as zero, discards writes.
 * Unused operand slots are filled with it rather than left clear, which is what
 * the vendor compiler does and what the hardware expects. */
#define HP_RZ 255
/* Predicate 7 is PT, the always-true predicate. An unpredicated instruction is
 * really an instruction predicated on PT. */
#define HP_PT 7

/* Place `value` (width `bits`) at bit position `pos` in a 128-bit word.
 * Positions and widths are compile-time constants at every call site, so this
 * folds away entirely at -O2. */
static inline void hp_put(hp_word *w, unsigned pos, unsigned bits, uint64_t value) {
  const uint64_t mask = (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
  const uint64_t v = value & mask;

  if (pos >= 64) {
    w->hi |= v << (pos - 64);
    return;
  }
  w->lo |= v << pos;
  /* A field may straddle the 64-bit boundary — the 32-bit source-B field at
   * bit 32 does not, but the control field's neighbours do, and getting this
   * wrong would silently truncate. */
  if (pos + bits > 64) w->hi |= v >> (64 - pos);
}

static inline uint64_t hp_get(const hp_word *w, unsigned pos, unsigned bits) {
  const uint64_t mask = (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
  if (pos >= 64) return (w->hi >> (pos - 64)) & mask;
  uint64_t v = w->lo >> pos;
  if (pos + bits > 64) v |= w->hi << (64 - pos);
  return v & mask;
}

static inline int hp_word_eq(hp_word a, hp_word b) {
  return a.lo == b.lo && a.hi == b.hi;
}

#endif /* HELIOS_HEPHAESTUS_WORD_H */
