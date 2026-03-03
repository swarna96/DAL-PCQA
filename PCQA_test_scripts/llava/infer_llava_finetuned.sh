#!/bin/bash
# Run inference with the finetuned LLaVA (LoRA adapter) on PCQA test set.
# By default uses the question from each sample in the JSON. Add --use_structured_prompt in the Python call to use the fixed PCQA prompt.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAVA_DATA="${LLAVA_DATA:-$ROOT/llava_data}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/dataset}"
cd "$SCRIPT_DIR"

ADAPTER_DIR="${1:-./checkpoints/llava_pcqa}"
OUTPUT="${2:-$ROOT/answers/llava_finetuned_pcqa_test.json}"
mkdir -p "$(dirname "$OUTPUT")"

python llava_zero_shot_infer.py \
  --test_json "$LLAVA_DATA/test_pcqa_llava.json" \
  --root_dir "$DATASET_ROOT" \
  --output "$OUTPUT" \
  --model_name "$ADAPTER_DIR" \
  --max_new_tokens 400 \
  --device 0

echo "Predictions saved to $OUTPUT (JSON: list of {id, text})"
