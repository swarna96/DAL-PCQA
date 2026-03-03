#!/bin/bash
# Fine-tune InternVL on PCQA with LoRA (for ~24GB GPU: batch_size=1, gradient_accumulation_steps=16).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAVA_DATA="${LLAVA_DATA:-$ROOT/llava_data}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/dataset}"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_NAME="${1:-OpenGVLab/InternVL3-8B-hf}"
OUTPUT_DIR="${2:-./checkpoints/internvl_pcqa}"

python internvl_finetune.py \
  --train_json "$LLAVA_DATA/train_pcqa_llava.json" \
  --root_dir "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --num_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --max_length 512 \
  --bf16 \
  --load_in_4bit \
  --gradient_checkpointing \
  --warmup_ratio 0.03 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 10

echo "Checkpoint saved to $OUTPUT_DIR"
echo "Inference: ./infer_internvl_finetuned.sh $OUTPUT_DIR"
