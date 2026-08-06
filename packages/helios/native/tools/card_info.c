/*
 * card_info.c — dump NV_ESC_CARD_INFO as the kernel actually fills it.
 *
 * WHAT: issues the enumeration ioctl and prints every 72-byte entry, so the
 * struct offsets we rely on can be checked against a real machine instead of
 * against a hand-computed alignment.
 *
 * WHY it exists: aether's enumeration read `minor_number` at offset 56 and got
 * a value that did not correspond to any device node. Either the offset is
 * wrong or the ioctl did not fill the buffer, and those two need different
 * fixes -- so the first job is to look.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no RM objects, no allocation. It opens
 * the control node, asks one question, and prints the answer.
 *
 * Build: gcc -O2 -o card_info tools/card_info.c aether/ioctl.c
 */
#include "../aether/nv_abi.h"
#include "../aether/ioctl.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define ENTRY 72

int main(void) {
  int fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
  if (fd < 0) { perror("/dev/nvidiactl"); return 1; }

  /* Try several buffer sizes: the ioctl request code encodes the size, so a
   * mismatch is rejected outright. NV_MAX_DEVICES has changed across driver
   * generations, and guessing wrong looks exactly like "no cards". */
  const int sizes[] = { ENTRY * 32, ENTRY * 64, ENTRY * 8, 4096 };
  static unsigned char buf[ENTRY * 64];

  for (unsigned s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
    memset(buf, 0, sizeof buf);
    int rc = aether_ioctl(fd, NV_ESC_CARD_INFO, buf, (size_t)sizes[s]);
    printf("CARD_INFO size=%d -> rc=%d\n", sizes[s], rc);
    if (rc != 0) continue;

    for (int e = 0; (e + 1) * ENTRY <= sizes[s]; e++) {
      const unsigned char *c = buf + e * ENTRY;
      int nonzero = 0;
      for (int i = 0; i < ENTRY; i++) if (c[i]) { nonzero = 1; break; }
      if (!nonzero) continue;

      printf("  entry %d: valid=%u gpu_id@16=0x%08x minor@56=%u name@60=%.10s\n",
             e, c[0], *(const unsigned *)(c + 16), *(const unsigned *)(c + 56),
             (const char *)(c + 60));
      printf("    raw:");
      for (int i = 0; i < ENTRY; i++) {
        if (i % 16 == 0) printf("\n      +%02d ", i);
        printf("%02x ", c[i]);
      }
      printf("\n");
    }
    break; /* first size that works is the one the kernel wants */
  }

  close(fd);
  return 0;
}
