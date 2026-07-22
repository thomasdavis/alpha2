#!/usr/bin/env bash
# Run one Llama-form 100M-token learning-rate pilot using the exact G3 contract.
# Invoke separately for 1e-3, 2e-3, and 3e-3 so each paid run is verified
# before the next is launched.

set -euo pipefail

learning_rate=${1:?learning rate required}
data=${2:?training text path required}
tokenizer=${3:?tokenizer artifact path required}
run_dir=${4:?run directory required}
resume_checkpoint=${5:-}

export ALPHA_PILOT_LR=$learning_rate
exec "$(dirname "$0")/run_g3_pilot.sh" llama "$data" "$tokenizer" "$run_dir" "$resume_checkpoint"
