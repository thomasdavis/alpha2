/**
 * build.mjs — Compile the Vulkan native addon from scratch.
 *
 * Zero dependencies. Uses gcc + Node.js headers (shipped with Node).
 * Output: native/helios_vk.node
 */

import { execSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = join(__dirname, "helios_vk.c");
const out = join(__dirname, "helios_vk.node");

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

const cmd = [
  "gcc",
  "-shared",
  "-fPIC",
  "-O3",
  "-DNDEBUG",
  "-flto",
  "-Wall",
  `-I${nodeDir}`,
  "-o", out,
  src,
  "-ldl",
].join(" ");

console.log(`Helios: compiling native addon...`);
console.log(`  ${cmd}`);

try {
  execSync(cmd, { stdio: "inherit", cwd: __dirname });
  console.log(`Helios: built ${out}`);
} catch (e) {
  console.error("Helios: native build failed");
  process.exit(1);
}
