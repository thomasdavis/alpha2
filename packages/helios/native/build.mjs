/**
 * build.mjs — Compile the Vulkan native addon from scratch.
 *
 * Zero dependencies. Uses gcc + Node.js headers (shipped with Node).
 * Output: native/helios_vk.node
 */

import { execSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync, statSync, readdirSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = join(__dirname, "helios_vk.c");
const out = join(__dirname, "helios_vk.node");
const script = fileURLToPath(import.meta.url);

/*
 * STALENESS IS DECIDED BY EVERY SOURCE, not by the translation unit's name.
 *
 * helios_vk.c is a single translation unit that #includes the whole tree, so
 * the compiler sees every .c and .h under here — but this check only ever
 * stat'ed helios_vk.c itself. Editing a kernel in prometheus/ therefore left
 * the addon "up-to-date" and the next run measured, and TESTED, the previous
 * binary. That is the most expensive failure mode this build has: a correctness
 * fix and a no-op are indistinguishable, and so are an optimisation and a
 * neutral one. The standing note in hmma.c about a failed build leaving the
 * previous addon in place is the same hazard arriving from the other side.
 *
 * Walking the tree costs a few milliseconds against a build that costs tens of
 * seconds, so there is no reason to be clever about which files can matter.
 */
function newestSourceMtime(dir) {
  let newest = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".") || entry.name === "node_modules") continue;
    const path = join(dir, entry.name);
    if (entry.isDirectory()) newest = Math.max(newest, newestSourceMtime(path));
    else if (/\.(c|h)$/.test(entry.name)) newest = Math.max(newest, statSync(path).mtimeMs);
  }
  return newest;
}

function isUpToDate() {
  if (!existsSync(out)) return false;
  const outMtime = statSync(out).mtimeMs;
  return outMtime >= newestSourceMtime(__dirname) && outMtime >= statSync(script).mtimeMs;
}

if (process.env.HELIOS_NATIVE_FORCE_REBUILD !== "1" && isUpToDate()) {
  console.log(`Helios: native addon up-to-date (${out})`);
  process.exit(0);
}

// Find Node.js include directory
const nodeDir = join(dirname(process.execPath), "..", "include", "node");
if (!existsSync(nodeDir)) {
  console.error(`Node.js headers not found at ${nodeDir}`);
  console.error("Install them with: sudo apt install libnode-dev");
  process.exit(1);
}

// Check for gcc
try {
  execSync("which gcc", { stdio: "pipe" });
} catch {
  console.error("gcc not found. Install it with: sudo apt install gcc");
  process.exit(1);
}

const isDarwin = process.platform === "darwin";

const cmdParts = [
  "gcc",
  "-shared",
  "-fPIC",
  "-O3",
  "-DNDEBUG",
  "-flto",
  "-Wall",
  `-I${nodeDir}`,
];

if (isDarwin) {
  cmdParts.push("-undefined", "dynamic_lookup");
}

cmdParts.push("-o", out, src);

if (!isDarwin) {
  cmdParts.push("-ldl");
}

const cmd = cmdParts.join(" ");

console.log(`Helios: compiling native addon...`);
console.log(`  ${cmd}`);

try {
  execSync(cmd, { stdio: "inherit", cwd: __dirname });
  console.log(`Helios: built ${out}`);
} catch (e) {
  console.error("Helios: native build failed");
  process.exit(1);
}
