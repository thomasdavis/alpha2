/*
 * pushbuffer_decode.c — reading a GPU method stream back into words.
 *
 * WHAT: walks a pushbuffer, decoding each header into an opcode, subchannel,
 * method address and operand count, and printing what it finds.
 *
 * WHY it is its own file: this is the half of the spy that has to be RIGHT
 * rather than merely useful. A walker that mis-decodes one opcode desynchronises
 * and every subsequent "header" is really a data word -- which is exactly what
 * happened, and it is why searching CUDA's traffic kept surfacing plausible
 * setup methods and never a launch. The bug was assuming every opcode is
 * followed by `count` data words; IMMD_DATA_METHOD has none, because its count
 * field IS the datum.
 */
#include "spy.h"

/* The compute methods worth naming when they appear. clc7c0.h. */
static const char *method_name(uint32_t m) {
  switch (m) {
    case 0x0000: return "SET_OBJECT";
    case 0x0180: return "LINE_LENGTH_IN";
    case 0x0188: return "OFFSET_OUT";
    case 0x01b0: return "LAUNCH_DMA";
    case 0x01b4: return "LOAD_INLINE_DATA";
    case 0x02b4: return "SEND_PCAS_A";
    case 0x02c0: return "SEND_SIGNALING_PCAS2_B";
    case 0x0318: return "SET_INLINE_QMD_ADDRESS_A";
    case 0x031c: return "SET_INLINE_QMD_ADDRESS_B";
    case 0x0320: return "LOAD_INLINE_QMD_DATA";
    default: return NULL;
  }
}

/*
 * Walk the method stream rather than hex-dumping it.
 *
 * A pushbuffer is self-describing, but only if the opcodes are decoded
 * correctly, and two details in clc56f.h make the naive version wrong:
 *
 *   NVC56F_DMA_METHOD_ADDRESS   11:0   (12:2 is _ADDRESS_OLD, a different era)
 *   NVC56F_DMA_SEC_OP           31:29
 *     0 GRP0_USE_TERT   2 GRP2_USE_TERT   6 RESERVED
 *     1 INC_METHOD      3 NON_INC_METHOD  5 ONE_INC    -> count data words
 *     4 IMMD_DATA_METHOD                  -> NO data words; the count FIELD
 *                                            (28:16) is itself the datum
 *     7 END_PB_SEGMENT                    -> stop
 *
 * The first version advanced by 1 + count for every opcode. One immediate
 * method -- and drivers emit them constantly, since any single-dword method is
 * cheaper that way -- desynchronises the walk, after which every "header" is
 * really a data word and the decode is noise. That is why walking CUDA's
 * pushbuffers surfaced plenty of plausible setup methods and never once reached
 * a launch: the walk was already lost by the time it got there.
 */
#define SEC_OP_GRP0 0u
#define SEC_OP_INC 1u
#define SEC_OP_GRP2 2u
#define SEC_OP_NON_INC 3u
#define SEC_OP_IMMD 4u
#define SEC_OP_ONE_INC 5u
#define SEC_OP_END_SEG 7u

/* Data words following a header, given its opcode. */
static int data_words(uint32_t op, uint32_t count) {
  switch (op) {
    case SEC_OP_INC:
    case SEC_OP_NON_INC:
    case SEC_OP_ONE_INC: return (int)count;
    case SEC_OP_IMMD: return 0; /* the count field IS the data */
    default: return -1;         /* unknown or end: stop walking */
  }
}
static void dump_pushbuffer(uint64_t addr, uint32_t dwords) {
  const uint32_t *p = (const uint32_t *)(uintptr_t)addr;
  if (dwords > 512) dwords = 512;
  fprintf(L, "    pushbuffer 0x%lx (%u dwords):\n", addr, dwords);

  for (uint32_t i = 0; i < dwords;) {
    const uint32_t h = p[i];
    const uint32_t op = h >> 29;
    const uint32_t count = (h >> 16) & 0x1fffu;
    const uint32_t sub = (h >> 13) & 7u;
    const uint32_t method = (h & 0xfffu) * 4;
    const int nd = data_words(op, count);
    if (nd < 0 || i + 1 + (uint32_t)nd > dwords) {
      fprintf(L, "      +0x%03x  %08x  op=%u (stop)\n", i * 4, h, op);
      break;
    }
    const char *nm = method_name(method);
    fprintf(L, "      +0x%03x  op=%u sub=%u method=0x%04x count=%-3u %s\n",
            i * 4, op, sub, method, count, nm ? nm : "");

    /*
     * The QMD arrives as an inline-to-memory upload, not as LOAD_INLINE_QMD_DATA.
     *
     * CUDA emits OFFSET_OUT (a 64-bit GPU address) + LINE_LENGTH_IN + LAUNCH_DMA
     * + LOAD_INLINE_DATA, which writes the payload into GPU memory through the
     * I2M path, and only then points SET_INLINE_QMD_ADDRESS_A at it. So the QMD
     * is carried as bulk data inside a DMA method, which is why watching for the
     * QMD-named methods found the address setter and never the contents.
     */
    if (method == 0x01b4 && count >= 0x20) {
      fprintf(L, "        *** INLINE DATA (%u dwords) ***\n", count);
      for (uint32_t k = 0; k < count; k += 8) {
        fprintf(L, "        +0x%02x ", k * 4);
        for (uint32_t q = 0; q < 8 && k + q < count; q++)
          fprintf(L, "%08x ", p[i + 1 + k + q]);
        fprintf(L, "\n");
      }
    } else if ((method == 0x0320 || method == 0x0318) && count >= 0x20) {
      fprintf(L, "        *** QMD (%u dwords) ***\n", count);
      for (uint32_t k = 0; k < count; k += 8) {
        fprintf(L, "        +0x%02x ", k * 4);
        for (uint32_t q = 0; q < 8 && k + q < count; q++)
          fprintf(L, "%08x ", p[i + 1 + k + q]);
        fprintf(L, "\n");
      }
    } else if (nd > 0 && nd <= 6) {
      fprintf(L, "        data:");
      for (int k = 0; k < nd; k++) fprintf(L, " %08x", p[i + 1 + k]);
      fprintf(L, "\n");
    }
    i += 1 + (uint32_t)nd;
  }
}

