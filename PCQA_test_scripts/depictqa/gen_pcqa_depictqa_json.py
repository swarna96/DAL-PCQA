#!/usr/bin/env python3
"""
Generate DepictQA training JSON from a PCQA annotations CSV and projection images.

Expects:
  - CSV with columns: Ply_name, Quality Description, Generated_Description (and optionally Label, MOS, etc.)
  - Projections under PROJECTIONS_DIR either as:
      PROJECTIONS_DIR/{ply_name}/0.png, .../1.png, .../2.png, .../3.png
    or:
      PROJECTIONS_DIR/{ply_name}.0.png, .../ply_name.1.png, .../ply_name.2.png, .../ply_name.3.png

Output JSON entries have task_type "quality_single_A_noref" and paths relative to PROJECTIONS_DIR
(use config.data.root_dir = PROJECTIONS_DIR when training).
"""

import argparse
import csv
import json
import os
import random
from typing import List, Optional


# Same prompt list as DepictQA detail task (gen_json_refA_detail.py)
QUALITY_QUESTIONS = [
    "Could you assess the overall quality of the point cloud and elaborate on your evaluation?",
    "How would you rate the point cloud's quality, and what factors contribute to your assessment?",
    "Can you provide a detailed evaluation of the point cloud's quality?",
    "Please evaluate the point cloud's quality and provide your reasons.",
    "How do you perceive the quality of the point cloud, and what aspects influence your judgment?",
    "Offer an assessment of the point cloud's quality, highlighting any strengths or weaknesses.",
    "What is your opinion on the quality of the point cloud? Explain your viewpoint.",
    "Assess the quality of the point cloud with detailed reasons.",
    "How does the point cloud's quality impact its overall effectiveness or appeal?",
    "Evaluate the point cloud's quality and justify your evaluation.",
    "How about the overall quality of the point cloud, and why?",
    "Provide a thorough evaluation of the point cloud's quality.",
    "Analyze the point cloud's quality, and detail your findings.",
    "Assess the point cloud's quality from a professional standpoint.",
    "How would you rate the overall quality of the point cloud, and why?",
    "What is your opinion on the point cloud's quality? Elaborate on your evaluation.",
    "Evaluate the quality of the point cloud and provide a comprehensive explanation.",
]


def find_projection_paths(projections_dir: str, ply_name: str) -> Optional[List[str]]:
    """
    Return list of 4 image paths (relative to projections_dir) for this point cloud.
    Supports:
      - projections_dir/ply_name/0.png, 1.png, 2.png, 3.png
      - projections_dir/ply_name.0.png, ply_name.1.png, ...
    Returns None if any of the 4 images is missing.
    """
    base = ply_name  # e.g. "bag_gQP_1_tQP_1.ply"
    folder_candidates = [os.path.join(base, f"{i}.png") for i in range(4)]
    all_exist = all(
        os.path.isfile(os.path.join(projections_dir, p)) for p in folder_candidates
    )
    if all_exist:
        return folder_candidates
    flat_candidates = [f"{base}.{i}.png" for i in range(4)]
    all_exist = all(
        os.path.isfile(os.path.join(projections_dir, p)) for p in flat_candidates
    )
    if all_exist:
        return flat_candidates
    return None


def get_description(row: dict, quality_col: str, generated_col: str) -> Optional[str]:
    """Prefer Generated_Description, fallback to Quality Description. Strip whitespace."""
    v = (row.get(generated_col) or row.get(quality_col) or "").strip()
    return v if v else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate DepictQA training JSON from PCQA CSV and projection images."
    )
    parser.add_argument(
        "annotations_csv",
        type=str,
        help="Path to CSV with columns: Ply_name, Quality Description, Generated_Description",
    )
    parser.add_argument(
        "projections_dir",
        type=str,
        help="Base directory of projection images (e.g. .../wpc_projections). "
        "Paths in JSON will be relative to this; set config.data.root_dir to this path.",
    )
    parser.add_argument(
        "output_json",
        type=str,
        help="Output path for the training JSON file (or output dir when --test-split-csv is used).",
    )
    parser.add_argument(
        "--test-split-csv",
        type=str,
        default=None,
        help="Path to test split CSV (e.g. test_1.csv) with 'name' column of ply filenames. "
        "When set, writes train_pcqa.json (train samples) and test_pcqa.json (test samples, inference format).",
    )
    parser.add_argument(
        "--test-csv-name-col",
        type=str,
        default="name",
        help="Column name for ply filename in test split CSV (default: name).",
    )
    parser.add_argument(
        "--one-per-ply",
        action="store_true",
        help="Emit one sample per point cloud (view 0 only). Default: one sample per view (4 per ply).",
    )
    parser.add_argument(
        "--quality-col",
        type=str,
        default="Quality Description",
        help="CSV column name for short quality description (default: Quality Description).",
    )
    parser.add_argument(
        "--generated-col",
        type=str,
        default="Generated_Description",
        help="CSV column name for long description (default: Generated_Description).",
    )
    parser.add_argument(
        "--ply-col",
        type=str,
        default="Ply_name",
        help="CSV column name for point cloud filename (default: Ply_name).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=131,
        help="Random seed for question sampling (default: 131).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    projections_dir = os.path.expanduser(args.projections_dir)
    if not os.path.isdir(projections_dir):
        raise FileNotFoundError(f"Projections directory not found: {projections_dir}")

    rows = []
    with open(args.annotations_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        for row in reader:
            rows.append(row)

    ply_col = args.ply_col
    quality_col = args.quality_col
    generated_col = args.generated_col

    samples = []
    skipped_no_desc = 0
    skipped_no_images = 0

    for row in rows:
        ply_name = (row.get(ply_col) or "").strip()
        if not ply_name:
            continue
        description = get_description(row, quality_col, generated_col)
        if not description:
            skipped_no_desc += 1
            continue

        paths = find_projection_paths(projections_dir, ply_name)
        if paths is None:
            skipped_no_images += 1
            continue

        views_to_emit = [paths[0]] if args.one_per_ply else paths
        question = random.choice(QUALITY_QUESTIONS)

        for rel_path in views_to_emit:
            sample = {
                "ply_name": ply_name,
                "image_ref": None,
                "image_A": rel_path,
                "image_B": None,
                "task_type": "quality_single_A_noref",
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": description},
                ],
            }
            samples.append(sample)

    test_ply_names = set()
    if args.test_split_csv:
        test_csv_path = os.path.expanduser(args.test_split_csv)
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Test split CSV not found: {test_csv_path}")
        with open(test_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            name_col = args.test_csv_name_col
            for row in reader:
                ply = (row.get(name_col) or "").strip()
                if ply:
                    test_ply_names.add(ply)
        print(f"  Loaded {len(test_ply_names)} ply names from test split: {test_csv_path}")

    out_dir = os.path.dirname(os.path.abspath(args.output_json)) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.test_split_csv and test_ply_names:
        train_samples = []
        test_samples = []
        for s in samples:
            ply = s["ply_name"]
            if ply in test_ply_names:
                test_samples.append(s)
            else:
                train_samples.append(s)

        for s in train_samples:
            del s["ply_name"]
        for s in test_samples:
            del s["ply_name"]

        train_path = os.path.join(out_dir, "train_pcqa.json")
        test_path = os.path.join(out_dir, "test_pcqa.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(train_samples)} train samples to {train_path}")

        test_infer = []
        for i, s in enumerate(test_samples):
            query = s["conversations"][0]["value"]
            test_infer.append({
                "id": f"pcqa_test_{i}",
                "image_ref": s["image_ref"],
                "image_A": s["image_A"],
                "image_B": s["image_B"],
                "query": query,
            })
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_infer, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(test_infer)} test samples to {test_path}")
        print("  Set config.data.root_dir to the projections dir; put these JSON files there (or under a subdir and set meta paths accordingly).")
    else:
        for s in samples:
            del s["ply_name"]
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(samples)} samples to {args.output_json}")

    if skipped_no_desc:
        print(f"  Skipped {skipped_no_desc} rows (no description).")
    if skipped_no_images:
        print(f"  Skipped {skipped_no_images} rows (missing projection images).")
    print("  Set config.data.root_dir to the projections_dir path when training.")


if __name__ == "__main__":
    main()
