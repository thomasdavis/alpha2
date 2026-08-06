/*
 * harness.c — counters and main() for the C test binaries.
 *
 * WHAT: the four globals harness.h refers to, plus a main() that calls ht_run()
 * and reports.
 *
 * WHY: kept separate from harness.h so every test binary links one copy of the
 * counters rather than each translation unit carrying its own.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no test discovery. Each binary has exactly
 * one ht_run(), which makes the link graph the test selector — a layer's test
 * binary links only that layer and the ones below it, so an upward dependency
 * fails to link rather than passing quietly (standard 8).
 */
#include "harness.h"

int ht_failures = 0;
int ht_checks = 0;
int ht_case_failed = 0;
const char *ht_current = "(none)";

int main(void) {
  ht_run();

  printf("\n  %d checks, %d failure%s\n", ht_checks, ht_failures,
         ht_failures == 1 ? "" : "s");

  /* A test binary that asserted nothing is a failure, not a pass. X60's first
   * suite reported green while silently exercising a fallback path; a zero
   * check count is the cheapest possible detector for that class of mistake. */
  if (ht_checks == 0) {
    printf("  FAIL: no checks ran — the suite tested nothing\n");
    return 1;
  }
  return ht_failures == 0 ? 0 : 1;
}
