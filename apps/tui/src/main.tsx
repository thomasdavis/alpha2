import React from "react";
import { render } from "ink";
import { App } from "./App.js";
import * as path from "node:path";
import * as fs from "node:fs";

// Resolve outputs dir: explicit arg, or walk up to find it
function findOutputsDir(): string {
  const arg = process.argv[2];
  if (arg) return path.resolve(arg);

  // Try common locations
  const candidates = [
    path.resolve("outputs"),
    path.resolve("../../outputs"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return path.resolve("outputs");
}

// Enter alternate screen buffer for fullscreen TUI
const isRawSupported = typeof process.stdin.setRawMode === "function";
if (isRawSupported) {
  process.stdout.write("\x1b[?1049h"); // enter alt screen
  process.stdout.write("\x1b[H");      // cursor to top-left
}

const outputsDir = findOutputsDir();
const instance = render(<App outputsDir={outputsDir} />, {
  patchConsole: true,
});

// Clean up alt screen on exit
function cleanup() {
  if (isRawSupported) {
    process.stdout.write("\x1b[?1049l"); // leave alt screen
  }
}

instance.waitUntilExit().then(cleanup);
process.on("exit", cleanup);
