#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
env_file=${ALPHA2_DISCORD_ENV_FILE:-$repo_root/.env.discord.local}
attestation=${1:-}
message_file=${2:-}

if [[ $attestation != --qualitative-improvement && $attestation != --meaningful-performance ]] || [[ -z $message_file ]]; then
  echo "usage: scripts/post_discord_progress.sh (--qualitative-improvement|--meaningful-performance) MESSAGE_FILE" >&2
  echo "Discord is reserved for controlled output improvements or measured meaningful performance findings, not routine status." >&2
  exit 2
fi

[[ -f $env_file ]] || { echo "Discord environment file not found: $env_file" >&2; exit 2; }
[[ -f $message_file ]] || { echo "message file not found: $message_file" >&2; exit 2; }

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

webhook_url=${ALPHA2_DISCORD_WEBHOOK_URL:-${DISCORD_WEBHOOK_URL:-}}
: "${webhook_url:?ALPHA2_DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL is required}"

message_bytes=$(wc -c < "$message_file")
(( message_bytes <= 1900 )) || {
  echo "message is $message_bytes bytes; split it below the 1,900-byte safety limit" >&2
  exit 2
}

jq -Rs '{content: .}' < "$message_file" |
  curl --fail-with-body --silent --show-error \
    --header 'Content-Type: application/json' \
    --data-binary @- \
    --output /dev/null \
    "$webhook_url"

echo "Discord webhook accepted $(basename -- "$message_file") ($message_bytes bytes)"
