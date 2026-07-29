#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
env_file=${ALPHA2_DISCORD_ENV_FILE:-$repo_root/.env.discord.local}
message_file=${1:?usage: scripts/post_discord_progress.sh MESSAGE_FILE}

[[ -f $env_file ]] || { echo "Discord environment file not found: $env_file" >&2; exit 2; }
[[ -f $message_file ]] || { echo "message file not found: $message_file" >&2; exit 2; }

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

: "${ALPHA2_DISCORD_WEBHOOK_URL:?ALPHA2_DISCORD_WEBHOOK_URL is required}"

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
    "$ALPHA2_DISCORD_WEBHOOK_URL"

echo "Discord webhook accepted $(basename -- "$message_file") ($message_bytes bytes)"
