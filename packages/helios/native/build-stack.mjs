#!/usr/bin/env node
/*
 * build-stack.mjs — the entire build system for the from-scratch GPU stack.
 *
 * WHAT: globs the layer directories, compiles each layer's test binary, and
 * (later) links the napi addon.
 *
 * WHY: the stack takes no dependencies, and that includes build tooling. There
 * is no cmake, no make, nothing to install — node and gcc, which the RunPod
 * bootstrap already provides.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no incremental builds, no dependency
 * scanning. The tree is small and gcc is fast; correctness of the link graph
 * matters far more than rebuild speed, because the link graph is what enforces
 * layering (standard 8).
 *
 * Layering: LAYERS is ordered bottom-up. A layer's test binary links that layer
 * and everything BELOW it, never above. So if gaia/ starts calling into hermes/,
 * its test binary fails to link — the architecture is checked by the compiler
 * rather than by review.
 *
 * Usage:
 *   node build-stack.mjs           build and run every layer's tests
 *   node build-stack.mjs aether    just that layer
 */
import { execFileSync } from "node:child_process";
import { readdirSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, ".build");

/* Bottom-up. The index in this array IS the dependency rule. */
const LAYERS = [
  "aether",      // ioctl transport
  "gaia",        // memory and address space
  "hermes",      // channels, pushbuffer, launch
  "chronos",     // fences and timeline
  "hephaestus",  // SASS assembler
  "prometheus",  // kernel IR and codegen
  "helios",      // facade
];

const CFLAGS = [
  "-std=gnu11",   /* O_CLOEXEC and friends are POSIX-but-not-C11 */
  "-O2",
  "-g",
  "-Wall",
  "-Wextra",
  "-Werror",
  /* Padding in the RM ABI structs is deliberate and load-bearing; we assert the
   * offsets in tests rather than letting the compiler repack anything. */
  "-Wno-unused-parameter",
];

const sources = (dir) => {
  const p = join(HERE, dir);
  if (!existsSync(p)) return [];
  return readdirSync(p)
    .filter((f) => f.endsWith(".c"))
    .map((f) => join(p, f));
};

function build(layer, upto) {
  /* Link this layer plus every layer below it — never above. */
  const libs = LAYERS.slice(0, upto + 1).flatMap(sources);
  /*
   * A layer's tests may span more than one file: `<layer>_test.c` is the entry
   * point and `<layer>_*_test.c` are additional ones it calls into. Splitting
   * them is what keeps any single file under the line cap without merging
   * unrelated encodings into one function, and the entry point still owns the
   * order things run in.
   */
  const test = join(HERE, "test", `${layer}_test.c`);
  if (!existsSync(test)) return null;
  const extraTests = readdirSync(join(HERE, "test"))
    .filter((f) => f.startsWith(`${layer}_`) && f.endsWith("_test.c") &&
                   f !== `${layer}_test.c`)
    .map((f) => join(HERE, "test", f));

  const harness = join(HERE, "test", "harness.c");
  const bin = join(OUT, `${layer}_test`);
  mkdirSync(OUT, { recursive: true });

  /* -lm: the checkers evaluate exp2f/log2f/sqrtf on the host as an independent
   * oracle. Deliberately not reimplemented — a hand-rolled expected value is a
   * second implementation, and two implementations agreeing proves nothing. */
  execFileSync("gcc", [...CFLAGS, "-o", bin, test, ...extraTests, harness, ...libs, "-lm"], {
    stdio: "inherit",
  });
  return bin;
}

const only = process.argv[2];
let ran = 0;
let failed = 0;

/*
 * The dev tools, compiled but not run.
 *
 * They are LD_PRELOAD libraries and standalone probes -- they need a live
 * driver and a CUDA process to do anything, so there is nothing to assert here.
 * What there IS to catch is that they still BUILD. qmd_spy had drifted out of
 * compiling entirely: spy.h declared spy_data_words while the definition was
 * static in another file under a different name, and nothing noticed because
 * nothing built it. A tool that does not compile is not a tool.
 */
function buildTools() {
  const dir = join(HERE, "tools");
  const srcs = readdirSync(dir).filter((f) => f.endsWith(".c"));
  /* Every spy shares spy.h and the pieces behind it, so they link as one
   * library rather than as separate objects that would each need their own
   * copy of the region table. */
  const shared = ["spy_memory.c", "pushbuffer_decode.c"].map((f) => join(dir, f));
  const preload = ["qmd_spy.c", "rm_spy.c"].filter((f) => srcs.includes(f));
  for (const f of preload) {
    execFileSync("gcc", ["-std=gnu11", "-fPIC", "-shared", "-O1", "-o",
                         join(OUT, f.replace(".c", ".so")), join(dir, f),
                         ...shared, "-ldl", "-lpthread"], { stdio: "inherit" });
  }
  const standalone = srcs.filter(
    (f) => !preload.includes(f) && !shared.some((s) => s.endsWith(f)));
  for (const f of standalone) {
    execFileSync("gcc", [...CFLAGS, "-o", join(OUT, f.replace(".c", "")),
                         join(dir, f), ...sources("aether"), ...sources("gaia"),
                         ...sources("hermes"), ...sources("hephaestus"),
                         ...sources("prometheus"), "-lm"], { stdio: "inherit" });
  }
  console.log(`  tools: ${preload.length + standalone.length} built`);
}

for (let i = 0; i < LAYERS.length; i++) {
  const layer = LAYERS[i];
  if (only && layer !== only) continue;

  let bin;
  try {
    bin = build(layer, i);
  } catch {
    console.error(`\n${layer}: COMPILE FAILED`);
    failed++;
    continue;
  }
  if (!bin) continue; // layer has no tests yet

  ran++;
  try {
    execFileSync(bin, [], { stdio: "inherit" });
  } catch {
    failed++;
  }
}

if (ran === 0) {
  console.error("no test binaries built — nothing was verified");
  process.exit(1);
}
buildTools();

console.log(`\n${ran} layer suite(s), ${failed} failing`);
process.exit(failed === 0 ? 0 : 1);
