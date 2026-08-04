/*
 * harness.h — the whole test framework.
 *
 * WHAT: assertion macros and a main() shim for the C test binaries.
 *
 * WHY: the from-scratch GPU stack takes no dependencies, and that includes test
 * frameworks. This is deliberately tiny — if it grows past a page it has stopped
 * being a harness and started being a dependency.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no fixtures, no mocking, no parameterised
 * suites, no parallelism. Tests here are straight-line C that either computes a
 * known answer or does not.
 *
 * On known answers (docs/FROM-SCRATCH-GPU-STACK.md, standard 5): expected values
 * must come from algebra or the hardware spec, never from a second
 * implementation. Two implementations agreeing tells you nothing when both are
 * wrong, which is exactly how X58's halved gradient norm and X60's broken
 * softmax both survived a full parity suite.
 */
#ifndef HELIOS_TEST_HARNESS_H
#define HELIOS_TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

extern int ht_failures;
extern int ht_checks;
extern const char *ht_current;

/* Names the test currently running, so a failure reports where it happened. */
#define HT_CASE(name)                                                          \
  do {                                                                         \
    ht_current = (name);                                                       \
    printf("  %-56s", (name));                                                 \
    fflush(stdout);                                                            \
  } while (0)

/* Closes the current case. Prints ok only if nothing failed inside it. */
#define HT_END()                                                               \
  do {                                                                         \
    printf("%s\n", ht_case_failed ? "" : "ok");                                \
    ht_case_failed = 0;                                                        \
  } while (0)

extern int ht_case_failed;

#define HT_FAIL(fmt, ...)                                                      \
  do {                                                                         \
    if (!ht_case_failed) printf("FAIL\n");                                     \
    ht_case_failed = 1;                                                        \
    ht_failures++;                                                             \
    printf("      %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);       \
  } while (0)

#define HT_TRUE(cond)                                                          \
  do {                                                                         \
    ht_checks++;                                                               \
    if (!(cond)) HT_FAIL("expected true: %s", #cond);                          \
  } while (0)

#define HT_EQ_U64(got, want)                                                   \
  do {                                                                         \
    ht_checks++;                                                               \
    uint64_t _g = (uint64_t)(got), _w = (uint64_t)(want);                      \
    if (_g != _w)                                                              \
      HT_FAIL("%s: got %" PRIu64 " (0x%" PRIx64 "), want %" PRIu64             \
              " (0x%" PRIx64 ")",                                              \
              #got, _g, _g, _w, _w);                                           \
  } while (0)

#define HT_EQ_PTR(got, want)                                                   \
  do {                                                                         \
    ht_checks++;                                                               \
    const void *_g = (const void *)(got), *_w = (const void *)(want);          \
    if (_g != _w) HT_FAIL("%s: got %p, want %p", #got, _g, _w);                \
  } while (0)

/* Struct layout checks. The RM ioctl structs are an ABI we do not control:
 * a wrong offset is not a compile error, it is a kernel rejecting our call or,
 * worse, reading the wrong field. Every struct we define gets these. */
#define HT_OFFSET(type, field, want)                                           \
  do {                                                                         \
    ht_checks++;                                                               \
    size_t _o = offsetof(type, field);                                         \
    if (_o != (size_t)(want))                                                  \
      HT_FAIL("offsetof(%s, %s) = %zu, want %d", #type, #field, _o,            \
              (int)(want));                                                    \
  } while (0)

#define HT_SIZEOF(type, want)                                                  \
  do {                                                                         \
    ht_checks++;                                                               \
    size_t _s = sizeof(type);                                                  \
    if (_s != (size_t)(want))                                                  \
      HT_FAIL("sizeof(%s) = %zu, want %d", #type, _s, (int)(want));            \
  } while (0)

/* Each test file defines this. */
void ht_run(void);

#endif /* HELIOS_TEST_HARNESS_H */
