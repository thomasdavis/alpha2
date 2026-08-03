#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT="/mnt/donto-data/donto-resources/research/alpha-helios-reimagined"
readonly PUBLIC_ROOT="/srv/alpha-research"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 SOURCE.md [SOURCE.md ...]" >&2
  exit 2
fi

for source_name in "$@"; do
  source_path="$SOURCE_ROOT/$source_name"
  if [[ ! -f "$source_path" ]]; then
    echo "missing source: $source_path" >&2
    exit 1
  fi

  base_name="$(basename "$source_name" .md)"
  public_markdown="$PUBLIC_ROOT/$base_name.md"
  public_html="$PUBLIC_ROOT/$base_name.html"
  page_title="$(sed -n 's/^# //p' "$source_path" | head -n 1)"
  if [[ -z "$page_title" ]]; then
    page_title="$base_name"
  fi

  install -m 0664 "$source_path" "$public_markdown"
  header_path="$(mktemp)"
  printf '%s\n' \
    '<header class="site">' \
    '  <span><a href="/research/">alpha.donto.org / research</a></span>' \
    "  <span>source: <a href=\"/research/$base_name.md\">$base_name.md</a></span>" \
    '</header>' > "$header_path"

  pandoc -f gfm-tex_math_dollars -t html5 --standalone \
    --toc --toc-depth=3 \
    --css /research/style.css \
    --metadata title="$page_title" \
    --include-before-body="$header_path" \
    -o "$public_html" "$public_markdown"
  chmod 0664 "$public_html"
  rm -f "$header_path"
  echo "published $public_html"
done
