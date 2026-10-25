#!/bin/bash
PROJECT_DIR=$(python3 -c 'from paths import PROJECT_ROOT; print(PROJECT_ROOT)')
echo "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/outputs/logs"

podman run -d --rm \
    --network host \
    --env-file "$PROJECT_DIR/.env" \
    -v "$PROJECT_DIR/outputs/logs:/outputs/logs:Z" \
    --name inference-client \
    --requires lakefs \
    inference-client