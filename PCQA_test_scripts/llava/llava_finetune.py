#!/usr/bin/env python3
"""
Fine-tune LLaVA (HuggingFace) on PCQA LLaVA-format training data.

Reads train_pcqa_llava.json, builds labels (train only on assistant replies),
and runs HuggingFace Trainer. Saves checkpoints and final model to --output_dir.

Usage:
  pip install 'transformers>=4.36' accelerate torch pillow
  # For --load_in_4bit (LoRA): pip install bitsandbytes peft

  python llava_finetune.py \\
    --train_json llava_data/train_pcqa_llava.json \\
    --root_dir /path/to/dataset \\
    --output_dir ./checkpoints/llava_pcqa \\
    --model_name llava-hf/llava-1.5-7b-hf \\
    --num_epochs 3 \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 8 \\
    --bf16 \\
    --max_length 512
"""

import argparse
import json
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from transformers import (
        AutoProcessor,
        LlavaForConditionalGeneration,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
except ImportError as e:
    raise ImportError(
        "LLaVA finetune requires transformers>=4.36. Upgrade with: pip install 'transformers>=4.36'"
    ) from e

def _optional_peft():
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        return LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        return None, None, None


def _strip_image_tag(text: str) -> str:
    return re.sub(r"^\s*<image>\s*\n?", "", text).strip()


# LLaVA vision encoder (CLIP ViT) patch size and expected feature count from pixel grid.
LLAVA_PATCH_SIZE = 14


def _align_image_tokens_to_vision(input_ids, attention_mask, labels, pixel_values, image_token_id):
    """Ensure input_ids has exactly as many image tokens as the vision encoder will output for pixel_values.
    Returns (input_ids, attention_mask, labels), and the delta to add to prompt_len (for label masking).
    """
    # Vision encoder output size from pixel_values [C, H, W]
    h, w = pixel_values.shape[1], pixel_values.shape[2]
    n_features = (h // LLAVA_PATCH_SIZE) * (w // LLAVA_PATCH_SIZE)
    is_img = (input_ids == image_token_id)
    n_tokens = is_img.sum().item()
    if n_tokens == n_features:
        return input_ids, attention_mask, labels, 0
    # Find contiguous image-token span and replace with n_features tokens.
    idx = torch.nonzero(is_img, as_tuple=False)
    if idx.numel() == 0:
        return input_ids, attention_mask, labels, 0
    first, last = idx[0].item(), idx[-1].item()
    extra = n_features - (last - first + 1)
    new_input = torch.cat([
        input_ids[:first],
        torch.full((n_features,), image_token_id, dtype=input_ids.dtype, device=input_ids.device),
        input_ids[last + 1:],
    ])
    new_attn = torch.cat([
        attention_mask[:first],
        torch.ones(n_features, dtype=attention_mask.dtype, device=attention_mask.device),
        attention_mask[last + 1:],
    ])
    # Labels: same structure; image positions are masked with -100 so we just extend the masked region
    new_labels = torch.cat([
        labels[:first],
        torch.full((n_features,), -100, dtype=labels.dtype, device=labels.device),
        labels[last + 1:],
    ])
    return new_input, new_attn, new_labels, extra


class LLaVAPCQADataset(Dataset):
    """Dataset that loads LLaVA-format JSON and returns tokenized (image, text) with labels (assistant-only)."""

    def __init__(self, data_path: str, root_dir: str, processor, max_length: int = 512):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.processor = processor
        self.max_length = max_length
        self._image_token_id = getattr(processor, "image_token_id", None)
        if self._image_token_id is None and hasattr(processor.tokenizer, "convert_tokens_to_ids"):
            self._image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        if self._image_token_id is None:
            self._image_token_id = 32000  # LLaVA default
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
        # Resize to fixed size so the processor's image_processor yields consistent patch count.
        image = image.resize((336, 336), Image.Resampling.BILINEAR)
        conversation_prompt = [
            {"role": "user", "content": [{"type": "text", "text": s["user_text"]}, {"type": "image"}]}
        ]
        prompt_str = self.processor.apply_chat_template(
            conversation_prompt, add_generation_prompt=True, tokenize=False
        )
        if isinstance(prompt_str, list):
            prompt_str = prompt_str[0] if prompt_str else ""
        full_text = prompt_str + s["assistant_text"]
        # Do not truncate in processor: it can drop image tokens and cause token/feature count mismatch.
        inputs_full = self.processor(
            images=image, text=full_text, return_tensors="pt", padding=False, truncation=False
        )
        inputs_prompt = self.processor(
            images=image, text=prompt_str, return_tensors="pt", padding=False, truncation=False
        )
        prompt_len = inputs_prompt["input_ids"].shape[1]
        input_ids = inputs_full["input_ids"].squeeze(0)
        attention_mask = inputs_full["attention_mask"].squeeze(0)
        pixel_values = inputs_full["pixel_values"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        # Align image token count to vision encoder output size (fixes processor/model mismatch e.g. 507 vs 576).
        input_ids, attention_mask, labels, prompt_delta = _align_image_tokens_to_vision(
            input_ids, attention_mask, labels, pixel_values, self._image_token_id
        )
        prompt_len += prompt_delta
        labels[:prompt_len] = -100
        # Truncate from the right only; never cut image tokens (would cause token/feature count mismatch).
        n_features = (pixel_values.shape[1] // LLAVA_PATCH_SIZE) * (pixel_values.shape[2] // LLAVA_PATCH_SIZE)
        is_img = (input_ids == self._image_token_id)
        first_img = int(is_img.nonzero(as_tuple=False)[0].item()) if is_img.any() else 0
        min_keep = first_img + n_features
        keep_len = min(input_ids.shape[0], max(self.max_length, min_keep))
        if input_ids.shape[0] > keep_len:
            input_ids = input_ids[:keep_len].clone()
            attention_mask = attention_mask[:keep_len].clone()
            labels = labels[:keep_len].clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "pixel_values": pixel_values}


def collate_fn(examples, processor):
    # LLaVA forward expects n_image_tokens in input_ids to equal n_image_features from pixel_values.
    # With batch_size>1, different images can get different token counts from the processor, causing
    # "Image features and image tokens do not match". So we use batch_size=1 in practice; this
    # collator still supports batching if all samples happen to have the same image token count.
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
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA on PCQA")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/llava_pcqa")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Use 1 to avoid image token/feature count mismatch (LLaVA); use gradient_accumulation_steps for effective batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit and train LoRA adapters (requires bitsandbytes, peft)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (when using --load_in_4bit)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (when using --load_in_4bit)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (when using --load_in_4bit)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.expanduser(args.root_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading processor and model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    load_kwargs = dict(low_cpu_mem_usage=True)
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
    model = LlavaForConditionalGeneration.from_pretrained(args.model_name, **load_kwargs)
    if not getattr(model, "is_loaded_in_4bit", False):
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    if args.load_in_4bit:
        LoraConfig, get_peft_model, prepare_model_for_kbit_training = _optional_peft()
        if get_peft_model is None:
            raise ImportError("4-bit fine-tuning requires peft. Install with: pip install peft")
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
    train_dataset = LLaVAPCQADataset(args.train_json, root_dir, processor, args.max_length)
    print(f"Train samples: {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training samples loaded. Check that:\n"
            f"  1) --train_json exists and is LLaVA-format (image, conversations with human+gpt).\n"
            f"  2) --root_dir {root_dir!r} is correct: image paths in the JSON are relative to this dir.\n"
            f"  Check --train_json and --root_dir so image paths in the JSON resolve under root_dir."
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
        # With LoRA, use_reentrant=False so gradients flow to adapter (fixes "None of the inputs have requires_grad").
        gradient_checkpointing_kwargs={"use_reentrant": False} if (args.gradient_checkpointing and args.load_in_4bit) else None,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        data_collator=lambda ex: collate_fn(ex, processor),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model and processor saved to {args.output_dir}")
    if args.load_in_4bit:
        print("This is a LoRA adapter. For inference: load base model then merge/load adapter, or use peft AutoPeftModel.")
    print("Run inference with llava_zero_shot_infer.py using --model_name", args.output_dir)


if __name__ == "__main__":
    main()
