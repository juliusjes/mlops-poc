#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use first argument as dataset directory, or default to SCRIPT_DIR/data
DATASET_DIR="${1:-$SCRIPT_DIR/data}"
mkdir -p "$DATASET_DIR"

ZIP_FILE="$DATASET_DIR/wm811k-wafer-map.zip"

echo "Downloading dataset..."
curl -L -o "$ZIP_FILE" \
  https://www.kaggle.com/api/v1/datasets/download/qingyi/wm811k-wafer-map

echo "Extracting dataset..."
unzip -o "$ZIP_FILE" -d "$DATASET_DIR"

DATA_FILE="$DATASET_DIR/LSWMD.pkl"
if [ ! -f "$DATA_FILE" ]; then
  EXTRACTED_FILE="$(find "$DATASET_DIR" -type f -name LSWMD.pkl -print -quit)"
  if [ -z "$EXTRACTED_FILE" ]; then
    echo "LSWMD.pkl was not found in the downloaded archive." >&2
    exit 1
  fi
  mv "$EXTRACTED_FILE" "$DATA_FILE"
fi

echo "Done. Dataset is available at: $DATA_FILE"
