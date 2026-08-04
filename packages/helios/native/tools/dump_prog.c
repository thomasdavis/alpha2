/* Emit the test kernel to a raw file so nvdisasm can decode it. */
#include "../hephaestus/sm86.h"
#include <stdio.h>
int main(void) {
  hp_word p[5];
  p[0] = hp_mov_imm(0, 0x00060000u, hp_ctrl_safe());
  p[1] = hp_mov_imm(1, 0x00000008u, hp_ctrl_safe());
  p[2] = hp_mov_imm(2, 0xcafef00du, hp_ctrl_safe());
  p[3] = hp_stg(0, 2, 0, hp_ctrl_safe());
  p[4] = hp_exit(hp_ctrl_safe());
  FILE *f = fopen("/tmp/ours.bin", "wb");
  fwrite(p, sizeof p, 1, f);
  fclose(f);
  return 0;
}
