#!/usr/bin/env bash
set -euo pipefail

ATTEMPTS="${BUN_COMPILE_ATTEMPTS:-3}"
TIMEOUT_SEC="${BUN_COMPILE_TIMEOUT_SEC:-90}"
OUT_DIR="${BUN_COMPILE_OUT_DIR:-.bun-out}"
ENTRY="${BUN_COMPILE_ENTRY:-./apps/cli/dist/main.js}"
OUTFILE="${BUN_COMPILE_OUTFILE:-${OUT_DIR}/alpha}"

echo "[bun:compile] building TypeScript + native addon..."
npm run build -w @alpha/cli
npm run build:native -w @alpha/helios

mkdir -p "${OUT_DIR}"

run_compile() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "${TIMEOUT_SEC}" bun build --compile "${ENTRY}" --outfile "${OUTFILE}"
  else
    bun build --compile "${ENTRY}" --outfile "${OUTFILE}"
  fi
}

ok=0
for attempt in $(seq 1 "${ATTEMPTS}"); do
  echo "[bun:compile] attempt ${attempt}/${ATTEMPTS} (timeout=${TIMEOUT_SEC}s)"
  if run_compile; then
    ok=1
    break
  fi
  echo "[bun:compile] attempt ${attempt} failed"
  sleep 1
done

if [[ "${ok}" != "1" ]]; then
  echo "[bun:compile] failed after ${ATTEMPTS} attempt(s)" >&2
  exit 1
fi

cp packages/helios/native/helios_vk.node "${OUT_DIR}/helios_vk.node"
echo "[bun:compile] done: ${OUTFILE}"
