#!/usr/bin/env python3
"""
Zero-shot inference with LLaVA (HuggingFace) on PCQA test set.

Reads LLaVA-format test JSON (e.g. llava_data/test_pcqa_llava.json), runs the model
on each image + question, and saves predictions as JSON: a list of {"id": "...", "text": "..."}.


  python llava_zero_shot_infer.py \\
    --test_json llava_data/test_pcqa_llava.json \\
    --root_dir /path/to/dataset \\
    --output answers/llava_zero_shot_pcqa_test.json \\
    --model_name llava-hf/llava-1.5-7b-hf \\
    --max_new_tokens 400 \\
    --device 0

  By default the question is taken from each sample in the JSON (conversations[0].value).
  To use a single structured prompt for all samples:
    --use_structured_prompt
  To force using JSON prompts explicitly:
    --use_json_prompts
  Or pass a custom prompt for all samples:
    --prompt "Analyze this point cloud..."

  Finetuned (LoRA) checkpoint: point --model_name to the adapter directory (e.g. checkpoints/llava_pcqa).
  The script will load the base model and adapter automatically. Optional: --load_in_4bit to save VRAM.
"""

import argparse
import json
import os
import re

import torch
from PIL import Image
from tqdm import tqdm

# LLaVA requires transformers >= 4.36
try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
except ImportError as e:
    raise ImportError(
        "LLaVA requires transformers>=4.36. Upgrade with: pip install 'transformers>=4.36'"
    ) from e


def _is_peft_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))

# Optional structured prompt for consistent, detailed PCQA analysis (use --use_structured_prompt)
STRUCTURED_PCQA_PROMPT = """Analyze this 3D point cloud view step by step:
1. Examine the overall structure and geometry.
2. Look for noise, blurriness, or artifacts in the point distribution.
3. Check texture, color, and brightness consistency.
4. Note any missing regions or deformed shape.
Based on these observations, provide a brief analysis of the distortions and classify the quality as one of: Excellent, Good, Fair, Poor, or Bad.
Answer format: "Analysis: [observations]. Distortions: [summary]. Quality: [Excellent/Good/Fair/Poor/Bad]" """


def main():
    parser = argparse.ArgumentParser(description="LLaVA zero-shot inference on PCQA test set")
    parser.add_argument("--test_json", type=str, required=True, help="LLaVA-format test JSON")
    parser.add_argument("--root_dir", type=str, required=True, help="Root dir for image paths in JSON")
    parser.add_argument("--output", type=str, default="answers/llava_zero_shot_pcqa_test.json")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization (saves VRAM)")
    parser.add_argument("--use_structured_prompt", action="store_true", help="Use built-in step-by-step PCQA prompt for all samples")
    parser.add_argument("--use_json_prompts", action="store_true", help="Use the question from each sample in the JSON (default if no other prompt option is set)")
    parser.add_argument("--prompt", type=str, default=None, help="Use this prompt for all samples (overrides other prompt options)")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.expanduser(args.root_dir))
    with open(args.test_json, encoding="utf-8") as f:
        test_data = json.load(f)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    load_kwargs = dict(
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float16

    is_adapter = os.path.isdir(args.model_name) and _is_peft_adapter(args.model_name)
    if is_adapter:
        with open(os.path.join(args.model_name, "adapter_config.json"), encoding="utf-8") as f:
            adapter_config = json.load(f)
        base_name = adapter_config.get("base_model_name_or_path", "llava-hf/llava-1.5-7b-hf")
        print(f"Loading base model {base_name} and LoRA adapter from {args.model_name} ...")
        model = LlavaForConditionalGeneration.from_pretrained(base_name, **load_kwargs)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_name, is_trainable=False)
        if not getattr(model, "is_loaded_in_4bit", False):
            model = model.to(device)
        else:
            device = next(model.parameters()).device
        processor = AutoProcessor.from_pretrained(args.model_name)
    else:
        print(f"Loading model {args.model_name} ...")
        model = LlavaForConditionalGeneration.from_pretrained(args.model_name, **load_kwargs)
        if not getattr(model, "is_loaded_in_4bit", False):
            model = model.to(device)
        else:
            device = next(model.parameters()).device
        processor = AutoProcessor.from_pretrained(args.model_name)

    results = []
    for item in tqdm(test_data, desc="Inference"):
        sample_id = item.get("id", "")
        image_rel = item.get("image", "")
        if not image_rel:
            results.append({"id": sample_id, "text": ""})
            continue
        image_path = os.path.join(root_dir, image_rel)
        if not os.path.isfile(image_path):
            results.append({"id": sample_id, "text": f"[Error: image not found {image_path}]"})
            continue

        # Get question: --prompt > --use_structured_prompt > from JSON (default or --use_json_prompts)
        if args.prompt:
            question = args.prompt.strip()
        elif args.use_structured_prompt:
            question = STRUCTURED_PCQA_PROMPT.strip()
        else:
            # Use the question from this sample in the JSON (default behavior)
            convs = item.get("conversations", [])
            if not convs or convs[0].get("from") != "human":
                results.append({"id": sample_id, "text": ""})
                continue
            raw_prompt = convs[0].get("value", "")
            question = re.sub(r"^\s*<image>\s*\n?", "", raw_prompt).strip() or "Describe the quality of this point cloud."

        # LLaVA chat format: user content = [text, image]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            results.append({"id": sample_id, "text": f"[Error loading image: {e}]"})
            continue

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        # Move to device; input_ids/attention_mask must stay long (int), only float tensors use model dtype
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to(device)
                if hasattr(v, "dtype") and v.dtype.is_floating_point and hasattr(model, "dtype") and model.dtype is not None:
                    v = v.to(model.dtype)
                inputs[k] = v

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )

        # Decode only the generated part (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        generated = output[0][input_len:]
        text = processor.decode(generated, skip_special_tokens=True).strip()
        results.append({"id": sample_id, "text": text})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} predictions to {args.output}")
    print("Predictions format: list of {id, text}. Use this path with your evaluation pipeline.")


if __name__ == "__main__":
    main()
