#!/usr/bin/env node
import { runCli } from "./cli.js";

runCli().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
