#!/bin/bash
# Fine-tune LLaVA on PCQA 
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAVA_DATA="${LLAVA_DATA:-$ROOT/llava_data}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/dataset}"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python llava_finetune.py \
  --train_json "$LLAVA_DATA/train_pcqa_llava.json" \
  --root_dir "$DATASET_ROOT" \
  --output_dir ./checkpoints/llava_pcqa \
  --model_name llava-hf/llava-1.5-7b-hf \
  --num_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --max_length 768 \
  --bf16 \
  --load_in_4bit \
  --gradient_checkpointing \
  --warmup_ratio 0.03 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 10
