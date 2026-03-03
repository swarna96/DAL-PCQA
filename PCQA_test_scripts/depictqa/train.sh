#!/bin/bash
# Train DepictQA on PCQA. Runs DepictQA's src/train.py with config in this folder.
# Usage: export DEPICTQA_ROOT=/path/to/DepictQA-main; ./train.sh <gpu_id>

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/config.yaml"

if [ -z "${DEPICTQA_ROOT}" ]; then
  echo "Error: DEPICTQA_ROOT is not set. Set it to the DepictQA repo root, e.g.:"
  echo "  export DEPICTQA_ROOT=/path/to/DepictQA-main"
  exit 1
fi
if [ ! -d "$DEPICTQA_ROOT/src" ]; then
  echo "Error: DEPICTQA_ROOT does not contain src/ (got: $DEPICTQA_ROOT)"
  exit 1
fi

export PYTHONPATH="$DEPICTQA_ROOT/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${1:-0}"

cd "$DEPICTQA_ROOT" || exit 1
deepspeed --include localhost:${1:-0} --master_addr 127.0.0.1 --master_port 28501 \
  "$DEPICTQA_ROOT/src/train.py" --cfg "$CONFIG_PATH"
