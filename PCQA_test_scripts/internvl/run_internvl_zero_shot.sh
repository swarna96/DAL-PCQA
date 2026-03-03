#!/bin/bash
# InternVL zero-shot inference on PCQA test set.
# Requires: pip install 'transformers>=4.37' accelerate torch pillow
# Optional: pip install bitsandbytes then add --load_in_4bit for less VRAM.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAVA_DATA="${LLAVA_DATA:-$ROOT/llava_data}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/dataset}"
cd "$SCRIPT_DIR"

TEST_JSON="${1:-$LLAVA_DATA/test_pcqa_llava.json}"
OUTPUT="${2:-$ROOT/answers/internvl_zero_shot_pcqa_test.json}"
MODEL="${3:-OpenGVLab/InternVL3-8B-hf}"
mkdir -p "$(dirname "$OUTPUT")"

python internvl_zero_shot_infer.py \
  --test_json "$TEST_JSON" \
  --root_dir "$DATASET_ROOT" \
  --output "$OUTPUT" \
  --model_name "$MODEL" \
  --max_new_tokens 400 \
  --device 0 \
  --use_structured_prompt

echo "Predictions saved to $OUTPUT (JSON: list of {id, text})"
