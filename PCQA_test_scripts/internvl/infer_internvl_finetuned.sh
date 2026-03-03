#!/bin/bash
# Run inference with the finetuned InternVL (LoRA adapter) on PCQA test set.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAVA_DATA="${LLAVA_DATA:-$ROOT/llava_data}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/dataset}"
cd "$SCRIPT_DIR"

ADAPTER_DIR="${1:-./checkpoints/internvl_pcqa}"
OUTPUT="${2:-$ROOT/answers/internvl_finetuned_pcqa_test.json}"
mkdir -p "$(dirname "$OUTPUT")"

python internvl_zero_shot_infer.py \
  --test_json "$LLAVA_DATA/test_pcqa_llava.json" \
  --root_dir "$DATASET_ROOT" \
  --output "$OUTPUT" \
  --model_name "$ADAPTER_DIR" \
  --max_new_tokens 400 \
  --device 0

echo "Predictions saved to $OUTPUT (JSON: list of {id, text})"
