#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";

import {
  analyzeCooperativeMatrixCapabilities,
  canonicalizeCooperativeMatrixProperties,
} from "./cooperative-matrix-capabilities.js";
import { destroyDevice, getNativeAddonPath, initDevice } from "./device.js";

function sha256File(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function git(args: string[]): string {
  try {
    return execFileSync("git", args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return "unavailable";
  }
}

const outputFlag = process.argv.indexOf("--output");
const outputPath = outputFlag >= 0 ? process.argv[outputFlag + 1] : undefined;
if (outputFlag >= 0 && !outputPath) {
  throw new Error("--output requires a path");
}

const info = initDevice();
try {
  const nativeAddonPath = getNativeAddonPath();
  const properties = canonicalizeCooperativeMatrixProperties(
    info.cooperativeMatrixProperties ?? [],
  );
  const stableCapabilityRecord = {
    schema: "alpha-helios-cooperative-matrix-capabilities-v1",
    implementation: {
      repositoryRevision: git(["rev-parse", "HEAD"]),
      repositoryDirty: git(["status", "--porcelain", "--untracked-files=all"]) !== "",
      nativeAddonSha256: sha256File(nativeAddonPath),
      nodeVersion: process.version,
      platform: process.platform,
      architecture: process.arch,
    },
    device: {
      deviceName: info.deviceName,
      vendorId: info.vendorId,
      deviceId: info.deviceId,
      deviceType: info.deviceType,
      apiVersion: info.apiVersion,
      driverVersion: info.driverVersion,
      subgroupSize: info.subgroupSize,
      f16Supported: info.f16Supported,
      coopMatSupported: info.coopMatSupported,
      coopMat2Supported: info.coopMat2Supported,
    },
    properties,
    analysis: analyzeCooperativeMatrixCapabilities(properties),
  };
  const canonical = JSON.stringify(stableCapabilityRecord);
  const output = {
    ...stableCapabilityRecord,
    capabilityFingerprintSha256: createHash("sha256").update(canonical).digest("hex"),
  };
  const rendered = `${JSON.stringify(output, null, 2)}\n`;
  if (outputPath) writeFileSync(outputPath, rendered, { encoding: "utf8", flag: "wx" });
  process.stdout.write(rendered);
} finally {
  destroyDevice();
}
