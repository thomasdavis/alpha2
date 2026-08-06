/*
 * control.c — see control.h.
 */
#include "control.h"

uint32_t hp_control_pack(hp_control c) {
  return (c.stall & 0xf) | ((c.yield & 0x1) << 4) |
         ((c.writeBarrier & 0x7) << 5) | ((c.readBarrier & 0x7) << 8) |
         ((c.waitMask & 0x3f) << 11) | ((c.reuse & 0xf) << 17);
}

hp_control hp_control_unpack(uint32_t p) {
  hp_control c;
  c.stall = p & 0xf;
  c.yield = (p >> 4) & 0x1;
  c.writeBarrier = (p >> 5) & 0x7;
  c.readBarrier = (p >> 8) & 0x7;
  c.waitMask = (p >> 11) & 0x3f;
  c.reuse = (p >> 17) & 0xf;
  return c;
}
