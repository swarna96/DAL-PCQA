#!/usr/bin/env python3
"""
Zero-shot inference with InternVL (HuggingFace) on PCQA test set.

Reads LLaVA-format test JSON (e.g. llava_data/test_pcqa_llava.json), runs the model
on each image + question, and saves predictions as JSON: a list of {"id": "...", "text": "..."}.

Requires: pip install 'transformers>=4.37' accelerate torch pillow
Optional: pip install bitsandbytes then add --load_in_4bit for less VRAM.

Usage:
  python internvl_zero_shot_infer.py \\
    --test_json llava_data/test_pcqa_llava.json \\
    --root_dir /path/to/dataset \\
    --output answers/internvl_zero_shot_pcqa_test.json \\
    --model_name OpenGVLab/InternVL2-8B \\
    --max_new_tokens 400 \\
    --device 0

For finetuned (LoRA) checkpoint: point --model_name to the adapter directory.
"""

import argparse
import json
import os
import re

import torch
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError as e:
    raise ImportError(
        "InternVL requires transformers>=4.37. Upgrade with: pip install 'transformers>=4.37'"
    ) from e


def _is_peft_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))

STRUCTURED_PCQA_PROMPT = """Analyze this 3D point cloud view step by step:
1. Examine the overall structure and geometry.
2. Look for noise, blurriness, or artifacts in the point distribution.
3. Check texture, color, and brightness consistency.
4. Note any missing regions or deformed shape.
Based on these observations, provide a brief analysis of the distortions and classify the quality as one of: Excellent, Good, Fair, Poor, or Bad.
Answer format: "Analysis: [observations]. Distortions: [summary]. Quality: [Excellent/Good/Fair/Poor/Bad]" """


def main():
    parser = argparse.ArgumentParser(description="InternVL zero-shot inference on PCQA test set")
    parser.add_argument("--test_json", type=str, required=True, help="LLaVA-format test JSON")
    parser.add_argument("--root_dir", type=str, required=True, help="Root dir for image paths in JSON")
    parser.add_argument("--output", type=str, default="answers/internvl_zero_shot_pcqa_test.json")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-8B-hf",
                        help="Use -hf (native) models: InternVL3-1B-hf, InternVL3-8B-hf. InternVL2-8B (custom) is not supported.")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization (saves VRAM)")
    parser.add_argument("--use_structured_prompt", action="store_true")
    parser.add_argument("--use_json_prompts", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.expanduser(args.root_dir))
    with open(args.test_json, encoding="utf-8") as f:
        test_data = json.load(f)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    load_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs["device_map"] = "auto"
        except ImportError:
            raise ImportError("4-bit requires: pip install bitsandbytes")
    else:
        load_kwargs["device_map"] = device if "cuda" in device else "auto"

    is_adapter = os.path.isdir(args.model_name) and _is_peft_adapter(args.model_name)
    if is_adapter:
        with open(os.path.join(args.model_name, "adapter_config.json"), encoding="utf-8") as f:
            adapter_config = json.load(f)
        base_name = adapter_config.get("base_model_name_or_path", "OpenGVLab/InternVL3-8B-hf")
        print(f"Loading base model {base_name} and LoRA adapter from {args.model_name} ...")
        model = AutoModelForImageTextToText.from_pretrained(base_name, **load_kwargs)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_name, is_trainable=False)
        if not getattr(model, "is_loaded_in_4bit", False):
            model = model.to(device)
        else:
            device = next(model.parameters()).device
        processor = AutoProcessor.from_pretrained(args.model_name)
    else:
        print(f"Loading model {args.model_name} ...")
        try:
            model = AutoModelForImageTextToText.from_pretrained(args.model_name, **load_kwargs)
        except ValueError as e:
            if "Unrecognized configuration" in str(e) or "InternVL" in str(e):
                raise ValueError(
                    f"Model {args.model_name} uses a custom config not in this transformers version. "
                    "Use a native -hf model instead: OpenGVLab/InternVL3-1B-hf or OpenGVLab/InternVL3-8B-hf, "
                    "and ensure pip install 'transformers>=4.52'."
                ) from e
            raise
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

        if args.prompt:
            question = args.prompt.strip()
        elif args.use_structured_prompt:
            question = STRUCTURED_PCQA_PROMPT.strip()
        else:
            convs = item.get("conversations", [])
            if not convs or convs[0].get("from") != "human":
                results.append({"id": sample_id, "text": ""})
                continue
            raw_prompt = convs[0].get("value", "")
            question = re.sub(r"^\s*<image>\s*\n?", "", raw_prompt).strip() or "Describe the quality of this point cloud."

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            results.append({"id": sample_id, "text": f"[Error loading image: {e}]"})
            continue

        # InternVL chat format: user content list with image and text.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if not isinstance(inputs, dict):
            if hasattr(inputs, "keys"):
                inputs = {k: v for k, v in inputs.items()}
            else:
                results.append({"id": sample_id, "text": "[Error: processor returned unexpected format]"})
                continue
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "to"):
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype if hasattr(model, "dtype") else torch.float16)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )

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
