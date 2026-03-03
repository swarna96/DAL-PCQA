#!/bin/bash
# LLM-as-a-judge: score generated PCQA descriptions vs reference (1–5).
# Run from PCQA_test_scripts (parent of eval/).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$(dirname "$SCRIPT_DIR")}"
cd "$ROOT" || exit 1

# Predictions JSON (id, text) from LLaVA / InternVL inference
PREDICTIONS="${1:-answers/llava_zero_shot_pcqa_test.json}"
OUT_JSON="${2:-answers/llm_judge_scores.json}"
# Test JSON and annotations (required for reference)
TEST_JSON="${3:-llava_data/test_pcqa_llava.json}"
ANNOTATIONS="${4:-}"  # e.g. path/to/generated_averaged_descriptions.csv or .xlsx

MODEL="${LLM_JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

echo "Predictions: $PREDICTIONS"
echo "Output: $OUT_JSON"
echo "Test JSON: $TEST_JSON"
echo "Annotations: $ANNOTATIONS"
echo "Model: $MODEL"

if [ -z "$ANNOTATIONS" ]; then
  echo "Usage: $0 [PREDICTIONS] [OUT_JSON] [TEST_JSON] ANNOTATIONS"
  echo "  ANNOTATIONS = path to CSV or XLSX with Ply_name and reference column (e.g. Generated_Description)"
  exit 1
fi

python eval/llm_judge_pcqa.py \
  --predictions "$PREDICTIONS" \
  --model "$MODEL" \
  --test_json "$TEST_JSON" \
  --annotations "$ANNOTATIONS" \
  --ref_col Generated_Description \
  --out_json "$OUT_JSON"

# Optional: 4-bit to save VRAM (pip install bitsandbytes)
# Add: --load_in_4bit

# Quick test on 10 samples:
# python eval/llm_judge_pcqa.py --predictions "$PREDICTIONS" --model "$MODEL" --test_json "$TEST_JSON" --annotations "$ANNOTATIONS" --max_samples 10 --out_json "$OUT_JSON"
