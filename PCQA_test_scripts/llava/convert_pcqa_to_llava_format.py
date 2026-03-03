#!/usr/bin/env python3
"""
Convert PCQA data to LLaVA-1.5 format using a descriptions file and a test split file.

- **Descriptions file** (e.g. generated_averaged_point_cloud_descriptions_SJTU_new2.csv): has ply name
  and description (e.g. Ply_name, Generated_Description). All samples with descriptions.
- **Test split file** (e.g. test_1.csv): lists which ply names are in the test set (e.g. column "name").
- **Train** = every ply in the descriptions file that is NOT in the test split → train_pcqa_llava.json
  (description taken from the descriptions file).
- **Test** = every ply in the test split file → test_pcqa_llava.json (no description needed).

LLaVA-1.5 expects:
  - "image": path (relative to root_dir)
  - "conversations": [{"from": "human", "value": "<image>\\nQuestion"}, {"from": "gpt", "value": "Answer"}]

Usage:
  python convert_pcqa_to_llava_format.py \\
    --descriptions_csv path/to/<annotated file> \\
    --test_split_csv path/to/sjtu_data_info/<test_split_file> \\
    --root_dir /path/to/dataset \\
    --out_dir ./llava_data \\
    --projections_subdir <projections_subdir> \\
    --ply_column Ply_name \\
    --description_column Generated_Description \\
    --test_ply_column name
"""

import argparse
import json
import os


def _load_table(path: str):
    """Load CSV or XLSX with pandas if available."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "CSV/XLSX input requires pandas. Install with: pip install pandas. For XLSX also: pip install openpyxl"
        ) from e
    path_lower = path.lower()
    if path_lower.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        return pd.read_excel(path, engine="openpyxl" if path_lower.endswith(".xlsx") else None)
    raise ValueError(f"Unsupported table format: {path}. Use .csv or .xlsx.")


def main():
    parser = argparse.ArgumentParser(
        description="Build LLaVA train/test JSON from a descriptions file and a test split file."
    )
    parser.add_argument("--descriptions_csv", type=str, default=None,
                        help="Path to CSV with ply name + description (e.g. generated_averaged_point_cloud_descriptions_SJTU_new2.csv)")
    parser.add_argument("--descriptions_xlsx", type=str, default=None, help="Path to XLSX (same structure)")
    parser.add_argument("--test_split_csv", type=str, default=None,
                        help="Path to CSV listing test ply names (e.g. test_1.csv with column name)")
    parser.add_argument("--test_split_xlsx", type=str, default=None, help="Path to XLSX for test split")
    parser.add_argument("--root_dir", type=str, required=True, help="Image root (paths in output stay relative to this)")
    parser.add_argument("--out_dir", type=str, default="./llava_data", help="Output directory")
    parser.add_argument("--projections_subdir", type=str, default="projections",
                        help="Subdir under root_dir for projection images. Image path: <this>/<ply_name>/<view_id>.png")
    parser.add_argument("--ply_column", type=str, default="Ply_name",
                        help="Column name for ply in descriptions file (e.g. Ply_name)")
    parser.add_argument("--description_column", type=str, default="Generated_Description",
                        help="Column name for description in descriptions file")
    parser.add_argument("--test_ply_column", type=str, default="name",
                        help="Column name for ply in test split file (e.g. name)")
    parser.add_argument("--question", type=str, default="Describe the quality of this point cloud.",
                        help="Fixed question for each sample")
    parser.add_argument("--view_id", type=int, default=0, help="Projection view index for image path")
    args = parser.parse_args()

    descriptions_path = args.descriptions_csv or args.descriptions_xlsx
    test_split_path = args.test_split_csv or args.test_split_xlsx

    if not descriptions_path:
        parser.error("Provide --descriptions_csv or --descriptions_xlsx (file with ply name + description).")
    if not test_split_path:
        parser.error("Provide --test_split_csv or --test_split_xlsx (file listing which ply names are in the test set).")
    if not os.path.isfile(descriptions_path):
        parser.error(f"Descriptions file not found: {descriptions_path}")
    if not os.path.isfile(test_split_path):
        parser.error(f"Test split file not found: {test_split_path}")

    root_dir = os.path.abspath(args.root_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    import pandas as pd

    # Load descriptions file: ply -> description
    df_desc = _load_table(descriptions_path)
    ply_col = args.ply_column
    desc_col = args.description_column
    if ply_col not in df_desc.columns:
        raise ValueError(f"Descriptions: ply column {args.ply_column!r} not found. Columns: {list(df_desc.columns)}")
    if desc_col not in df_desc.columns:
        raise ValueError(f"Descriptions: description column {args.description_column!r} not found. Columns: {list(df_desc.columns)}")

    ply_to_description = {}
    for idx, row in df_desc.iterrows():
        ply_name = row.get(ply_col)
        if pd.isna(ply_name) or not str(ply_name).strip():
            continue
        ply_name = str(ply_name).strip()
        desc = row.get(desc_col)
        if pd.isna(desc) or not str(desc).strip():
            continue
        ply_to_description[ply_name] = str(desc).strip()

    # Load test split: set of ply names that are test
    df_test_split = _load_table(test_split_path)
    test_ply_col = args.test_ply_column
    if test_ply_col not in df_test_split.columns:
        raise ValueError(f"Test split: ply column {args.test_ply_column!r} not found. Columns: {list(df_test_split.columns)}")

    test_ply_names = set()
    for _, row in df_test_split.iterrows():
        ply_name = row.get(test_ply_col)
        if pd.isna(ply_name) or not str(ply_name).strip():
            continue
        test_ply_names.add(str(ply_name).strip())

    # Train = descriptions file minus test set (use description from descriptions file)
    train_llava = []
    for idx, (ply_name, desc) in enumerate(ply_to_description.items()):
        if ply_name in test_ply_names:
            continue
        image_rel = os.path.join(args.projections_subdir, ply_name, f"{args.view_id}.png").replace("\\", "/")
        train_llava.append({
            "id": f"pcqa_train_{idx}",
            "image": image_rel,
            "conversations": [
                {"from": "human", "value": "<image>\n" + args.question},
                {"from": "gpt", "value": desc},
            ],
        })

    # Test = test split file (each row -> one test sample, no description)
    test_llava = []
    for idx, row in df_test_split.iterrows():
        ply_name = row.get(test_ply_col)
        if pd.isna(ply_name) or not str(ply_name).strip():
            continue
        ply_name = str(ply_name).strip()
        image_rel = os.path.join(args.projections_subdir, ply_name, f"{args.view_id}.png").replace("\\", "/")
        test_llava.append({
            "id": f"pcqa_test_{idx}",
            "image": image_rel,
            "conversations": [
                {"from": "human", "value": "<image>\n" + args.question},
                {"from": "gpt", "value": ""},
            ],
        })

    out_path = os.path.join(args.out_dir, "train_pcqa_llava.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(train_llava, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(train_llava)} train samples to {out_path}")

    out_path = os.path.join(args.out_dir, "test_pcqa_llava.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_llava, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(test_llava)} test samples to {out_path}")

    print(f"\nSet LLaVA data root to: {root_dir}")
    print("Use the JSON files under --out_dir for finetuning and inference.")


if __name__ == "__main__":
    main()
