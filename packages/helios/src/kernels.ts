/**
 * kernels.ts — Barrel re-export for backward compatibility.
 *
 * The actual kernel implementations live in kernels/ submodules.
 * This file re-exports everything so existing imports (e.g. backend.ts)
 * continue to work unchanged.
 */

export * from "./kernels/index.js";
