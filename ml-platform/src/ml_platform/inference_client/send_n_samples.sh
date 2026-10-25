#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <N> <UUID>"
    exit 1
fi

N="$1"
UUID="$2"

curl --fail-with-body -X POST "http://localhost:8013/send_n_samples" \
    -H "Content-Type: application/json" \
    -d "{
        \"n\": ${N},
        \"labels\": true,
        \"uuid\": \"${UUID}\"
    }"