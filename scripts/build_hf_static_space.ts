#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { renderUi } from "../apps/hf/src/ui.js";

const outputFlag = process.argv.indexOf("--out");
if (outputFlag < 0 || !process.argv[outputFlag + 1]) {
  throw new Error("usage: build_hf_static_space.ts --out /absolute/path/index.html");
}
const output = resolve(process.argv[outputFlag + 1]!);
if (!output.startsWith("/mnt/donto-data/alpha-runs/")) {
  throw new Error("--out must live under /mnt/donto-data/alpha-runs/");
}
await mkdir(dirname(output), { recursive: true });
await writeFile(output, renderUi(57_688_576, 1_200, "https://donto.org/alpha-60m"), {
  encoding: "utf-8",
  flag: "wx",
});
console.log(output);
