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
  const test = join(HERE, "test", `${layer}_test.c`);
  if (!existsSync(test)) return null;

  const harness = join(HERE, "test", "harness.c");
  const bin = join(OUT, `${layer}_test`);
  mkdirSync(OUT, { recursive: true });

  /* -lm: the checkers evaluate exp2f/log2f/sqrtf on the host as an independent
   * oracle. Deliberately not reimplemented — a hand-rolled expected value is a
   * second implementation, and two implementations agreeing proves nothing. */
  execFileSync("gcc", [...CFLAGS, "-o", bin, test, harness, ...libs, "-lm"], {
    stdio: "inherit",
  });
  return bin;
}

const only = process.argv[2];
let ran = 0;
let failed = 0;

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
console.log(`\n${ran} layer suite(s), ${failed} failing`);
process.exit(failed === 0 ? 0 : 1);
