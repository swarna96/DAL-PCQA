#!/usr/bin/env python3
"""
Fine-tune InternVL (HuggingFace) on PCQA LLaVA-format training data.

Reads train_*_pcqa_llava.json, builds labels (train only on assistant replies),
and runs HuggingFace Trainer with optional LoRA. Saves checkpoints to --output_dir.

Requires: pip install 'transformers>=4.37' accelerate torch pillow
For LoRA: pip install bitsandbytes peft

Usage:
  python internvl_finetune.py \\
    --train_json llava_data/train_pcqa_llava.json \\
    --root_dir /path/to/dataset \\
    --output_dir ./checkpoints/internvl_pcqa \\
    --model_name OpenGVLab/InternVL3-8B-hf \\
    --num_epochs 3 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 16 \\
    --bf16 \\
    --max_length 512 \\
    --load_in_4bit
"""

import argparse
import json
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset

# Workaround: peft may import HybridCache from transformers; transformers 5.2 doesn't export it.
import transformers
if not getattr(transformers, "HybridCache", None):
    try:
        from transformers.cache_utils import HybridCache
        transformers.HybridCache = HybridCache
    except ImportError:
        from transformers.cache_utils import DynamicCache
        transformers.HybridCache = DynamicCache

try:
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
except ImportError as e:
    raise ImportError(
        "InternVL finetune requires transformers>=4.37. pip install 'transformers>=4.37'"
    ) from e


def _optional_peft():
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        return LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        return None, None, None


def _strip_image_tag(text: str) -> str:
    return re.sub(r"^\s*<image>\s*\n?", "", text).strip()


class InternVLPCQADataset(Dataset):
    """Dataset that loads LLaVA-format JSON and returns tokenized (image, text) with labels (assistant-only)."""

    def __init__(self, data_path: str, root_dir: str, processor, max_length: int = 512):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.processor = processor
        self.max_length = max_length
        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.samples = []
        for item in raw:
            image_rel = item.get("image", "")
            convs = item.get("conversations", [])
            if not image_rel or len(convs) < 2 or convs[0].get("from") != "human" or convs[1].get("from") != "gpt":
                continue
            user_text = _strip_image_tag(convs[0].get("value", "")) or "Describe the quality."
            assistant_text = (convs[1].get("value", "") or "").strip()
            image_path = os.path.join(self.root_dir, image_rel)
            if not os.path.isfile(image_path):
                continue
            self.samples.append({"image_path": image_path, "user_text": user_text, "assistant_text": assistant_text})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            image = Image.open(s["image_path"]).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        image = image.resize((448, 448), Image.Resampling.BILINEAR)
        image_content = image

        messages_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_content},
                    {"type": "text", "text": s["user_text"]},
                ],
            }
        ]
        messages_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_content},
                    {"type": "text", "text": s["user_text"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": s["assistant_text"]}]},
        ]
        try:
            inputs_prompt = self.processor.apply_chat_template(
                messages_prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs_full = self.processor.apply_chat_template(
                messages_full,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as e:
            raise RuntimeError(f"apply_chat_template failed: {e}") from e

        if getattr(inputs_prompt, "get", None) is None or inputs_prompt.get("input_ids") is None:
            raise RuntimeError(
                f"apply_chat_template must return dict-like with input_ids; got {type(inputs_prompt).__name__}"
            )
        if getattr(inputs_full, "get", None) is None or inputs_full.get("input_ids") is None:
            raise RuntimeError(
                f"apply_chat_template must return dict-like with input_ids; got {type(inputs_full).__name__}"
            )

        prompt_len = inputs_prompt["input_ids"].shape[1]
        input_ids = inputs_full["input_ids"].squeeze(0)
        attention_mask = inputs_full.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        else:
            attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long()
        pixel_values = inputs_full.get("pixel_values")
        if pixel_values is None:
            pixel_values = inputs_prompt.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("Processor did not return pixel_values")
        pixel_values = pixel_values.squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length].clone()
            attention_mask = attention_mask[: self.max_length].clone()
            labels = labels[: self.max_length].clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }


def collate_fn(examples, processor):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [e["input_ids"] for e in examples], batch_first=True, padding_value=pad_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [e["attention_mask"] for e in examples], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [e["labels"] for e in examples], batch_first=True, padding_value=-100
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "pixel_values": pixel_values}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune InternVL on PCQA")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/internvl_pcqa")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-8B-hf",
                        help="Use -hf (native) models: InternVL3-1B-hf, InternVL3-8B-hf. InternVL2-8B is not supported.")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load in 4-bit and train LoRA (bitsandbytes, peft)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.expanduser(args.root_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading processor and model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    load_kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=True)
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model_name, **load_kwargs)
    except ValueError as e:
        if "Unrecognized configuration" in str(e) or "InternVL" in str(e):
            raise ValueError(
                f"Model {args.model_name} uses a custom config not supported by AutoModelForImageTextToText. "
                "Use a native -hf model: OpenGVLab/InternVL3-1B-hf or OpenGVLab/InternVL3-8B-hf."
            ) from e
        raise
    if not getattr(model, "is_loaded_in_4bit", False):
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    if args.load_in_4bit:
        LoraConfig, get_peft_model, prepare_model_for_kbit_training = _optional_peft()
        if get_peft_model is None:
            raise ImportError("4-bit fine-tuning requires peft. pip install peft")
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = InternVLPCQADataset(args.train_json, root_dir, processor, args.max_length)
    print(f"Train samples: {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training samples loaded. Check --train_json and --root_dir {root_dir!r}. "
            "Check --train_json and --root_dir so image paths in the JSON resolve under root_dir."
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        fp16=args.fp16,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False} if (args.gradient_checkpointing and args.load_in_4bit) else None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda ex: collate_fn(ex, processor),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model and processor saved to {args.output_dir}")
    if args.load_in_4bit:
        print("This is a LoRA adapter. For inference use internvl_zero_shot_infer.py with --model_name", args.output_dir)
    print("Run inference with internvl_zero_shot_infer.py using --model_name", args.output_dir)


if __name__ == "__main__":
    main()
