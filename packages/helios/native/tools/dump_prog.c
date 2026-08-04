/*
 * dump_prog.c — emit every instruction form Hephaestus can produce, so
 * nvdisasm can decode them.
 *
 * WHY this exists as a tool rather than a test: nvdisasm lives on the GPU box,
 * not in CI, and the round-trip it performs is the only check that catches an
 * encoding which is well-formed but means something else. Three bugs hid from
 * bit-comparison tests and all three were obvious the moment the output was
 * disassembled -- a MOV that read a constant bank instead of taking an
 * immediate, an STG missing its memory descriptor, and an EXIT that was
 * silently predicated.
 *
 * Usage on the box:
 *   gcc -o dump tools/dump_prog.c hephaestus/*.c && ./dump
 *   nvdisasm -b SM86 /tmp/ours.bin
 */
#include "../hephaestus/sm86.h"
#include <stdio.h>

int main(void) {
  hp_word p[] = {
      hp_mov_imm(0, 0x00060000u, hp_ctrl_safe()),
      hp_mov_imm(2, 0xcafef00du, hp_ctrl_safe()),
      hp_mov_const(1, 0, 0x28, hp_ctrl_safe()),
      hp_s2r(5, HP_SR_TID_X, hp_ctrl_setbar(0)),
      hp_iadd3_imm(7, 2, 0x1234, hp_ctrl_safe()),
      hp_ldg(3, 0, 0, hp_ctrl_setbar(0)),
      hp_ldg(4, 0, 0x40, hp_ctrl_setbar(0)),
      hp_stg(0, 2, 0, hp_ctrl_safe()),
      hp_stg(0, 2, 0x40, hp_ctrl_safe()),
      hp_bar_sync(hp_ctrl_safe()),
      hp_nop(hp_ctrl_safe()),
      hp_exit(hp_ctrl_safe()),
  };
  FILE *f = fopen("/tmp/ours.bin", "wb");
  fwrite(p, sizeof p, 1, f);
  fclose(f);
  return 0;
}
