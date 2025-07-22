#!/usr/bin/env bash

# Flags and arguments:
# --input_path : path to the raw phone data directory
# --capture_path : path to the capture directory (where files are saved)

if [ -z "$CAPTURE_DIR" ]; then
  echo "[ERROR] CAPTURE_DIR env var not set. Make sure to export CAPTURE_DIR=/path/to/data/root."
  exit 1
fi

if [ -z "$LOCATION" ]; then
  echo "[ERROR] LOCATION env var not set. Make sure to export LOCATION=location."
  exit 1
fi

CAPTURE="${CAPTURE_DIR}/${LOCATION}"
INPUT_DIR="${CAPTURE}/raw/phone"

echo "You are running with parameters: "
echo "  Capture: ${CAPTURE}"
echo "  Input path: ${INPUT_DIR}"

read -p "Do you want to continue? (y/n): " answer

if [[ ! "$answer" =~ ^[Yy]$ ]]; then
    echo "Execution aborted."
    exit 1
fi

echo "Running run_phone_to_capture on $LOCATION ..."

python3 -m scantools.run_phone_to_capture \
    --input_path $INPUT_DIR \
    --capture_path $CAPTURE \

echo "Done, run_phone_to_capture process completed on $LOCATION."
