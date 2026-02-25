#!/usr/bin/env bash
# Multipart upload to Vercel Blob using client token
# Usage: ./scripts/blob-upload.sh <file> <pathname> <client_token>
set -euo pipefail

FILE="$1"
PATHNAME="$2"
TOKEN="$3"
CHUNK_SIZE=$((50 * 1024 * 1024))  # 50MB chunks
BASE="https://blob.vercel-storage.com"

echo "Creating multipart upload for $PATHNAME..."
CREATE_RESP=$(curl -sf -X POST "${BASE}/${PATHNAME}?action=mpu-create" \
  -H "authorization: Bearer ${TOKEN}" \
  -H "x-api-version: 7")

UPLOAD_ID=$(echo "$CREATE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['uploadId'])")
echo "Upload ID: $UPLOAD_ID"

FILE_SIZE=$(stat -c%s "$FILE")
PART_NUM=1
OFFSET=0
PARTS_JSON="["

while [ $OFFSET -lt $FILE_SIZE ]; do
  REMAINING=$((FILE_SIZE - OFFSET))
  THIS_CHUNK=$CHUNK_SIZE
  if [ $REMAINING -lt $THIS_CHUNK ]; then
    THIS_CHUNK=$REMAINING
  fi

  echo "Uploading part $PART_NUM (offset=$OFFSET, size=$THIS_CHUNK)..."
  PART_RESP=$(dd if="$FILE" bs=1 skip=$OFFSET count=$THIS_CHUNK 2>/dev/null | \
    curl -sf -X PUT "${BASE}/${PATHNAME}?action=mpu-upload&uploadId=${UPLOAD_ID}&partNumber=${PART_NUM}" \
      -H "authorization: Bearer ${TOKEN}" \
      -H "x-api-version: 7" \
      -H "content-type: application/octet-stream" \
      --data-binary @-)

  ETAG=$(echo "$PART_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['etag'])")
  echo "  Part $PART_NUM uploaded, etag=$ETAG"

  if [ $PART_NUM -gt 1 ]; then
    PARTS_JSON="${PARTS_JSON},"
  fi
  PARTS_JSON="${PARTS_JSON}{\"partNumber\":${PART_NUM},\"etag\":\"${ETAG}\"}"

  OFFSET=$((OFFSET + THIS_CHUNK))
  PART_NUM=$((PART_NUM + 1))
done

PARTS_JSON="${PARTS_JSON}]"

echo "Completing multipart upload..."
COMPLETE_RESP=$(curl -sf -X POST "${BASE}/${PATHNAME}?action=mpu-complete" \
  -H "authorization: Bearer ${TOKEN}" \
  -H "x-api-version: 7" \
  -H "content-type: application/json" \
  -d "{\"uploadId\":\"${UPLOAD_ID}\",\"parts\":${PARTS_JSON}}")

echo "Upload complete!"
echo "$COMPLETE_RESP"
