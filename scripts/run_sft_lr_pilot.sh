#!/usr/bin/env bash
# Contracted 2,000-step SFT learning-rate pilot initialized from the final flagship base checkpoint.
# Arguments are identical to run_flagship_sft.sh; use one distinct run directory per sweep LR.

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ALPHA_SFT_JOB=lr-pilot exec "$script_dir/run_flagship_sft.sh" "$@"
