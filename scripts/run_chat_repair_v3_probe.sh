#!/usr/bin/env bash
# Execute the mandatory one-step, model-sized paired RCR-UL feasibility probe.
# This wraps the exact U1 launcher path with a contractually selection-ineligible
# mode; it must pass before either 400-step C0/U1 arm may begin.
#
# Usage:
#   scripts/run_chat_repair_v3_probe.sh \
#     POSITIVE_COHORT NEGATIVE_COHORT RCR_MANIFEST FREEZE_MANIFEST \
#     DEV_SFT TOKENIZER INIT_CHECKPOINT RUN_DIR

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $# -eq 8 ]] || {
  echo "usage: scripts/run_chat_repair_v3_probe.sh POSITIVE_COHORT NEGATIVE_COHORT RCR_MANIFEST FREEZE_MANIFEST DEV_SFT TOKENIZER INIT_CHECKPOINT RUN_DIR" >&2
  exit 2
}

ALPHA_V3_PROBE_ONLY=1 exec "$script_dir/run_chat_repair_v3_arm.sh" \
  U1 0.5 "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
