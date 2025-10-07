#!/usr/bin/env bash
set -e

CONFIG_FILE="./create_cross_valid/config.yaml"

CAPTURE_VALID_DIR=$(grep "cross_vdb_dir:" "$CONFIG_FILE" | awk '{print $2}')

if [ -d "$CAPTURE_VALID_DIR" ]; then
    echo "Removing existing directory: $CAPTURE_VALID_DIR"
    sudo rm -rf "$CAPTURE_VALID_DIR"
fi

echo "Running extract_capture..."
python3 -m create_cross_valid.create_captures --config "$CONFIG_FILE"