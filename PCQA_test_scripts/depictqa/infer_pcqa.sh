#!/bin/bash
# Run DepictQA inference on PCQA (quality_single_A_noref).
# Usage: export DEPICTQA_ROOT=/path/to/DepictQA-main; ./infer_pcqa.sh <gpu_id> [test_meta_path] [dataset_name]
# Example: ./infer_pcqa.sh 0 test_pcqa.json pcqa_test
# Test JSON must live under config.data.root_dir (or pass path that load_valset can resolve).

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

# Default test meta and output name (override with args 2 and 3)
TEST_META="${2:-test_pcqa.json}"
DATASET_NAME="${3:-pcqa_baseline}"

cd "$DEPICTQA_ROOT" || exit 1
python "$DEPICTQA_ROOT/src/infer.py" \
  --cfg "$CONFIG_PATH" \
  --meta_path "$TEST_META" \
  --dataset_name "$DATASET_NAME" \
  --task_name quality_single_A_noref \
  --batch_size 1

# Optional: use a fixed prompt for all samples (uncomment and set path):
#   --prompt "$(cat "$SCRIPT_DIR/structured_prompt_pcqa.txt")"
