#include "harness.h"
#include "../aether/ioctl.h"
#include "../hermes/pushbuffer.h"
#include "../prometheus/kernel.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../prometheus/kernel.h"

/* The device-side suite, in prometheus_hw_test.c. */
void pr_hardware_tests(void);

/*
 * The registry itself, checked without a GPU.
 *
 * WHY this exists: the hardware test skips where there is no device, and a
 * suite that skips everything runs no checks — which the harness now reports as
 * a failure, correctly. More usefully, these catch the mistakes that are easy
 * to make while ADDING a kernel: a missing checker, a builder that overruns the
 * instruction buffer, a launch geometry that does not cover the elements the
 * checker will read. All of those would otherwise surface as a confusing
 * hardware failure rather than as what they are.
 */
static void test_registry_is_wellformed(void) {
  HT_CASE("every kernel is completely specified");
  unsigned n = 0;
  const pr_kernel *ks = pr_kernels(&n);
  HT_TRUE(n > 0);

  for (unsigned i = 0; i < n; i++) {
    const pr_kernel *k = &ks[i];
    HT_TRUE(k->name != NULL && k->build != NULL && k->check != NULL);
    HT_TRUE(k->blockX > 0 && k->gridX > 0);

    /* The launch must cover every element a checker inspects, or the test
     * would be comparing against memory no thread ever wrote. */
    /* Threads times elements-per-thread must cover the tensor exactly. Checking
     * the PRODUCT rather than the thread count is what lets a kernel process a
     * pair per thread without the check either failing or being weakened to an
     * inequality that would stop catching a launch that covers too little. */
    const NvU32 perThread = k->elementsPerThread ? k->elementsPerThread : 1u;
    const NvU32 work = k->workElements ? k->workElements : PR_N;
    HT_EQ_U64(k->blockX * k->gridX * perThread, work);

    /*
     * And the builder must fit the buffer the runner gives it.
     *
     * Twice the bound, with a sentinel in the upper half: checking the returned
     * count alone happens after the damage, so it cannot catch an overrun. This
     * can.
     */
    hp_word prog[PR_MAX_INSTRUCTIONS * 2];
    const hp_word sentinel = {0xdeadbeefcafef00dull, 0x0123456789abcdefull};
    for (unsigned s = PR_MAX_INSTRUCTIONS; s < PR_MAX_INSTRUCTIONS * 2; s++)
      prog[s] = sentinel;
    const unsigned count = k->build(prog, 0x1000, 0x2000);
    HT_TRUE(count > 0 && count <= PR_MAX_INSTRUCTIONS);
    for (unsigned s = PR_MAX_INSTRUCTIONS; s < PR_MAX_INSTRUCTIONS * 2; s++)
      if (!hp_word_eq(prog[s], sentinel)) {
        HT_FAIL("%s overran its instruction buffer", k->name);
        break;
      }
  }
  HT_END();
}

/* Every kernel must end in EXIT. A program that runs off its own end is not a
 * kernel, and the padding that makes that survivable is a safety net rather
 * than a licence to rely on it. */
static void test_every_kernel_terminates(void) {
  HT_CASE("every kernel ends in EXIT");
  unsigned n = 0;
  const pr_kernel *ks = pr_kernels(&n);
  const hp_word exit_word = hp_exit(hp_ctrl_safe());
  for (unsigned i = 0; i < n; i++) {
    hp_word prog[PR_MAX_INSTRUCTIONS];
    const unsigned count = ks[i].build(prog, 0x1000, 0x2000);
    HT_TRUE(hp_word_eq(prog[count - 1], exit_word));
  }
  HT_END();
}

void ht_run(void) {
  printf("\nprometheus — kernels\n");
  test_registry_is_wellformed();
  test_every_kernel_terminates();
  pr_hardware_tests();
}
