/*
 * qmd_spy.c — recover the Ampere QMD layout from a running CUDA process.
 *
 * WHAT: an LD_PRELOAD library that scans a live CUDA process's GPU mappings for
 * a compute launch, then prints the method stream around it and the QMD it
 * points at.
 *
 * WHY this rather than a header: open-gpu-kernel-modules ships cla0c0qmd.h
 * (Kepler) and nothing for Ampere -- the QMD V03_00 field layout is not
 * published there. The alternative to reading it off real hardware is guessing
 * a bit-field layout from an older generation, which is exactly the class of
 * mistake that left GP_PUT at the wrong offset for days.
 *
 * HOW a launch is recognised. The launch sequence is
 *
 *   SEND_PCAS_A            (0x02b4)  QMD address >> 8
 *   SEND_PCAS_B            (0x02b8)
 *   SEND_SIGNALING_PCAS_B  (0x02bc)
 *   SEND_SIGNALING_PCAS2_B (0x02c0)  PCAS_ACTION
 *
 * four contiguous methods, so one INC_METHOD header at 0x2b4 with count 4. The
 * header's address field (bits 12:0) is 0x2b4 >> 2 = 0xad.
 *
 * WHY THE SIGNATURE ALONE IS NOT ENOUGH, learned the hard way: scanning every
 * writable region for that bit pattern produced three confident "launches"
 * whose surrounding words were plainly SASS. Thirteen bits of signature occurs
 * by chance in any large blob of compiled kernel code. The fix is not a wider
 * signature but a CONSEQUENCE CHECK -- a real launch's QMD address must itself
 * land inside a GPU mapping. The bogus hits pointed at 0x240f0200, which is
 * nowhere. Candidate by pattern, confirm by consequence.
 *
 * Identifying GPU mappings: CUDA maps its buffers at host addresses equal to
 * their GPU addresses (observed: mmap returns 0x200200000 for the object whose
 * gpFifoOffset is 0x200200000), which puts them in a distinctive band well
 * below the usual mmap area. Tracking provenance by interposing open()/mmap()
 * would be tighter, but libcuda reaches the device nodes through neither
 * symbol -- the hooks recorded nothing -- so the band plus the consequence
 * check is what actually works.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: it does not decode the QMD. A single dump
 * gives the bytes, not the meaning; naming fields needs several launches whose
 * grid sizes differ in known ways.
 *
 * On the soul constraint: a development tool, outside the training loop, in the
 * same category as nvdisasm. Nothing it produces is linked or shipped.
 *
 * Usage:
 *   gcc -shared -fPIC -O2 -o qmd_spy.so tools/qmd_spy.c -lpthread
 *   LD_PRELOAD=./qmd_spy.so python3 -c "import torch; ..."
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PCAS_A_ADDR (0x02b4u >> 2)      /* 0xad — SEND_PCAS_A */
#define INLINE_QMD_ADDR (0x0320u >> 2)  /* 0xc8 — LOAD_INLINE_QMD_DATA(0) */

/* The band CUDA maps GPU buffers into. Deliberately generous: precision comes
 * from the consequence check, not from this. */
#define GPU_VA_LO 0x100000000ull
#define GPU_VA_HI 0x1000000000ull

#define MAX_REGIONS 512
static struct { uint64_t lo, hi; } regions[MAX_REGIONS];
static int nregions;
static FILE *L;

static int in_gpu_region(uint64_t a, uint64_t len) {
  for (int i = 0; i < nregions; i++)
    if (a >= regions[i].lo && a + len <= regions[i].hi) return 1;
  return 0;
}

static void load_regions(void) {
  nregions = 0;
  FILE *m = fopen("/proc/self/maps", "r");
  if (!m) return;
  char line[512];
  while (nregions < MAX_REGIONS && fgets(line, sizeof line, m)) {
    uint64_t lo, hi;
    char perms[8];
    if (sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) != 3) continue;
    if (perms[0] != 'r' || perms[1] != 'w') continue;
    /*
     * GPU-visible regions only, and this restriction is not optional.
     *
     * Dropping it while hunting for the (wrong) PCAS signature seemed harmless;
     * re-running with the correct inline-QMD signature over ALL memory then
     * produced hits whose "QMD" was uniformly high-entropy -- random heap that
     * happens to match 13 bits of address and a plausible count. A pushbuffer
     * is by definition memory the GPU reads, so it is in the band. Signature
     * AND provenance; neither alone is enough.
     */
    if (lo < GPU_VA_LO || lo >= GPU_VA_HI) continue;
    if (hi - lo > (256u << 20)) continue;
    regions[nregions].lo = lo;
    regions[nregions].hi = hi;
    nregions++;
  }
  fclose(m);
}

/*
 * Read our own memory through /proc/self/mem rather than by dereferencing.
 *
 * Scanning every writable region directly killed the traced process: some of
 * those mappings fault on read (guard pages, device mappings with restricted
 * access), and a SIGSEGV inside the scanner thread takes the whole program with
 * it. pread on /proc/self/mem returns -1 for exactly those pages instead, which
 * turns an unreadable region into a skipped region.
 */
static int memfd = -1;

static long read_self(uint64_t addr, void *dst, size_t len) {
  if (memfd < 0) memfd = open("/proc/self/mem", O_RDONLY);
  if (memfd < 0) return -1;
  return pread(memfd, dst, len, (off_t)addr);
}

static void dump_qmd(uint64_t qmd) {
  uint32_t buf[64];
  if (read_self(qmd, buf, sizeof buf) != (long)sizeof buf) {
    fprintf(L, "    QMD @ 0x%lx unreadable\n", qmd);
    return;
  }
  const uint32_t *q = buf;
  fprintf(L, "    QMD @ 0x%lx:\n", qmd);
  for (int i = 0; i < 64; i += 8) {
    fprintf(L, "      +0x%02x ", i * 4);
    for (int k = 0; k < 8; k++) fprintf(L, "%08x ", q[i + k]);
    fprintf(L, "\n");
  }
}

#define CHUNK (1u << 20)
static uint32_t chunk[CHUNK / 4];

/*
 * FOLLOW THE RING instead of pattern-matching for a header.
 *
 * Every signature-based attempt here was either a false positive (random heap
 * matching 13 bits of method address) or a false negative (the right signature
 * in a region that was being skipped). Both failure modes come from guessing
 * where the data is.
 *
 * But the GPFIFO ring format is known exactly, verified against clc56f.h while
 * building our own submission path:
 *
 *   entry0  GET     31:2   pushbuffer address, low two bits are FETCH
 *   entry1  GET_HI   7:0   address bits 39:32
 *           LENGTH  30:10  length in dwords
 *
 * So: find the ring, walk its entries, and dump the method stream each one
 * points at. No signature, no guessing -- whatever launch encoding CUDA uses,
 * it is in there by construction.
 *
 * The ring is recognised structurally: a run of consecutive 8-byte entries whose
 * decoded addresses all land inside GPU-visible regions and whose lengths are
 * sane. That is a property no random data satisfies for long.
 */
static int plausible_entry(const uint32_t *e) {
  const uint64_t addr = ((uint64_t)(e[1] & 0xffu) << 32) | (e[0] & 0xfffffffcu);
  const uint32_t len = (e[1] >> 10) & 0x1fffffu;
  if (!addr || !len || len > 0x10000u) return 0;
  return in_gpu_region(addr, (uint64_t)len * 4);
}

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

/*
 * Every aligned 8-byte pair in a GPU region is a candidate GPFIFO entry.
 *
 * Locating "the ring" by looking for a run of consecutive valid entries found
 * only the FIRST channel's ring, and CUDA allocates nine channels whose rings
 * sit 0x3000 apart -- the compute one was never the one being read. There is no
 * need to identify rings at all: decode every pair, and keep the ones whose
 * pushbuffer actually contains a launch method. The pre-scan is the filter, and
 * it is a strong one, because a launch is a specific method address arrived at
 * by walking a self-describing stream rather than matched against raw bytes.
 */
static uint32_t hist[0x1000];
static int printed_init;

static int scan_once(void) {
  load_regions();
  int found = 0, seen = 0;
  for (int i = 0; i < nregions && found < 2; i++) {
    const uint32_t *w = (const uint32_t *)(uintptr_t)regions[i].lo;
    const uint64_t n = (regions[i].hi - regions[i].lo) / 4;
    for (uint64_t j = 0; j + 2 <= n && found < 2; j += 2) {
      if (!plausible_entry(&w[j])) continue;
      const uint64_t addr =
          ((uint64_t)(w[j + 1] & 0xffu) << 32) | (w[j] & 0xfffffffcu);
      uint32_t len = (w[j + 1] >> 10) & 0x1fffffu;
      if (len > 512) len = 512;

      const uint32_t *pb = (const uint32_t *)(uintptr_t)addr;
      int has_launch = 0;
      for (uint32_t k = 0; k < len;) {
        const uint32_t h = pb[k];
        const uint32_t op = h >> 29, cnt = (h >> 16) & 0x1fffu;
        const uint32_t m = (h & 0xfffu) * 4;
        const int nd = data_words(op, cnt);
        if (nd < 0 || k + 1 + (uint32_t)nd > len) break;
        if (m == 0x0318 || m == 0x0320 || m == 0x02b4) { has_launch = 1; break; }
        k += 1 + (uint32_t)nd;
      }
      seen++;
      /* Histogram every method seen across every pushbuffer. A launch either
       * appears in it or it does not, and that is a fact about the whole scan
       * rather than about the handful of samples that got printed. */
      for (uint32_t k = 0; k < len;) {
        const uint32_t h = pb[k];
        const uint32_t op = h >> 29, cnt = (h >> 16) & 0x1fffu;
        const int nd = data_words(op, cnt);
        if (nd < 0 || k + 1 + (uint32_t)nd > len) break;
        const uint32_t m = (h & 0xfffu) * 4;
        if (m < 0x4000) hist[m / 4]++;
        k += 1 + (uint32_t)nd;
      }
      if (!has_launch) {
        /* The compute-engine INIT pushbuffer is the one that begins with
         * SET_OBJECT, and it is as important as the launch: a launch into an
         * uninitialised engine raises GR_EXCEPTION. Print it whole. */
        if (!printed_init && len > 8 && (pb[0] & 0xfffu) == 0 &&
            (pb[0] >> 29) == 1u && pb[1] == 0xc7c0u) {
          fprintf(L, "\nCOMPUTE INIT pushbuffer 0x%lx (%u dwords)\n", addr, len);
          dump_pushbuffer(addr, len);
          printed_init = 1;
        }
        continue;
      }

      fprintf(L, "\nLAUNCH: entry at 0x%lx -> pushbuffer 0x%lx (%u dwords)\n",
              regions[i].lo + j * 4, addr, len);
      dump_pushbuffer(addr, len);
      found++;
    }
  }
  if (seen) {
    fprintf(L, "\n(%d pushbuffers walked, %d launches) methods seen:\n", seen,
            found);
    for (uint32_t m = 0; m < 0x1000; m++) {
      if (!hist[m]) continue;
      const char *nm = method_name(m * 4);
      fprintf(L, "   0x%04x x%-6u %s\n", m * 4, hist[m], nm ? nm : "");
    }
  }
  return found;
}

static void *scanner(void *unused) {
  (void)unused;
  fprintf(L, "scanner running\n");
  int peak = 0;
  for (int i = 0; i < 600; i++) {
    if (scan_once()) { fprintf(L, "\n-- done --\n"); return NULL; }
    if (nregions > peak) {
      peak = nregions;
      fprintf(L, "pass %d: %d GPU regions", i, nregions);
      for (int r = 0; r < nregions && r < 6; r++)
        fprintf(L, "  [0x%lx,0x%lx)", regions[r].lo, regions[r].hi);
      fprintf(L, "\n");
    }
    struct timespec ts = { 0, 20 * 1000 * 1000 };
    nanosleep(&ts, NULL);
  }
  fprintf(L, "no launch found (last pass saw %d GPU regions)\n", nregions);
  return NULL;
}

/*
 * The thread is created on the first getenv() call rather than in the
 * constructor.
 *
 * pthread_create from an LD_PRELOAD constructor runs before the threading
 * runtime is ready: it neither starts the thread nor reports an error, and the
 * log simply stays empty. Hooking a symbol the process is certain to call
 * afterwards gives a start point that is late enough to be safe and early
 * enough to catch the first launch.
 */
static pthread_once_t once = PTHREAD_ONCE_INIT;
static void spawn(void) {
  pthread_t t;
  if (pthread_create(&t, NULL, scanner, NULL) == 0) pthread_detach(t);
  else fprintf(L, "pthread_create failed\n");
}

char *getenv(const char *name) {
  extern char **environ;
  if (L) pthread_once(&once, spawn);
  if (!environ || !name) return NULL;
  const size_t n = strlen(name);
  for (char **e = environ; *e; e++)
    if (strncmp(*e, name, n) == 0 && (*e)[n] == '=') return *e + n + 1;
  return NULL;
}

__attribute__((constructor)) static void start(void) {
  L = fopen("/root/qmd.log", "w");
  if (L) setvbuf(L, NULL, _IONBF, 0);
}
