#!/usr/bin/env python3
"""
Evaluate predictions on the 5 quality labels: Excellent, Good, Fair, Poor, Bad.

Predictions JSON: list of {"id", "text"} (e.g. from LLaVA/InternVL inference).
Test JSON: list of {"id", "image_A"} or {"id", "image"} (LLaVA test format).
Annotations: CSV or XLSX with Ply_name, Label (and optionally MOS).

Usage:
  python eval_pcqa_labels.py \\
    --predictions ../answers/llava_zero_shot_pcqa_test.json \\
    --test_json ../llava_data/test_pcqa_llava.json \\
    --annotations path/to/descriptions.csv
  # Or Excel:
  python eval_pcqa_labels.py ... --annotations path/to/descriptions.xlsx
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = spearmanr = None

LABELS = ["Excellent", "Good", "Fair", "Poor", "Bad"]
LABEL_TO_SCORE = {lab: (5 - i) for i, lab in enumerate(LABELS)}


def _image_path_to_ply(image_path: str) -> str:
    """Extract ply name from image path (e.g. prefix/ply_name/0.png or ply_name/0.png)."""
    parts = (image_path or "").replace("\\", "/").split("/")
    for p in parts:
        if p.endswith(".ply"):
            return p
    if "/" in image_path:
        return parts[0]
    if image_path.endswith(".png"):
        return image_path.rsplit(".", 2)[0]
    return image_path or ""


def majority_vote(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = [lbl for lbl, c in counts.items() if c == max_count]
    return min(tied, key=lambda l: LABELS.index(l)) if tied else None


def label_to_score(label: Optional[str]) -> Optional[float]:
    if not label:
        return None
    for lab, score in LABEL_TO_SCORE.items():
        if lab.lower() == label.lower():
            return float(score)
    return None


def parse_predicted_label(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    text_lower = text.lower().strip()
    for label in LABELS:
        if f"rated as {label.lower()}" in text_lower:
            return label
    for label in LABELS:
        if label.lower() in text_lower:
            return label
    return None


def _load_annotations_rows(path: str) -> List[Dict[str, Any]]:
    """Load annotations from CSV or XLSX; return list of row dicts (str keys, str values)."""
    path_lower = path.lower()
    if path_lower.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("XLSX support requires pandas and openpyxl. pip install pandas openpyxl")
        df = pd.read_excel(path, engine="openpyxl" if path_lower.endswith(".xlsx") else None)
        return [
            {str(k): "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip()
             for k, v in row.items()}
            for row in df.to_dict("records")
        ]
    raise ValueError(f"Annotations file must be .csv or .xlsx, got: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5-label accuracy on PCQA predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to inference output JSON (id, text)")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test JSON (id, image_A or image)")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Annotations file: CSV or XLSX with Ply_name, Label (and optionally MOS)")
    parser.add_argument("--ply_col", type=str, default="Ply_name")
    parser.add_argument("--label_col", type=str, default="Label")
    parser.add_argument("--mos_col", type=str, default="MOS", help="CSV column for MOS (for PLCC/SRCC)")
    parser.add_argument("--out_ply_json", type=str, default=None, help="Optional: save per-ply pred/GT to JSON")
    args = parser.parse_args()

    with open(args.test_json, encoding="utf-8") as f:
        test_data = json.load(f)
    id_to_image = {}
    for item in test_data:
        pid = item.get("id")
        path = item.get("image_A") or item.get("image")
        if pid is not None and path:
            id_to_image[pid] = path

    ply_to_label = {}
    ply_to_mos = {}
    for row in _load_annotations_rows(args.annotations):
        ply = (row.get(args.ply_col) or "").strip()
        label = (row.get(args.label_col) or "").strip()
        mos_s = (row.get(args.mos_col) or "").strip()
        if ply and label:
            ply_to_label[ply] = label.title() if label.lower() in [l.lower() for l in LABELS] else label
        if ply and mos_s:
            try:
                ply_to_mos[ply] = float(mos_s)
            except ValueError:
                pass

    with open(args.predictions, encoding="utf-8") as f:
        pred_data = json.load(f)

    # Aggregate by ply (one GT label per ply; collect per-view predictions for majority vote)
    ply_to_gt: Dict[str, str] = {}
    ply_to_preds: Dict[str, List[Optional[str]]] = defaultdict(list)
    for pred in pred_data:
        pid = pred.get("id")
        text = pred.get("text", "")
        if pid not in id_to_image:
            continue
        image_path = id_to_image[pid]
        ply_name = _image_path_to_ply(image_path)
        gt_label = ply_to_label.get(ply_name)
        if gt_label is None:
            continue
        pred_label = parse_predicted_label(text)
        qalign_score = pred.get("qalign_score")
        if qalign_score is not None and pred_label is None and pred.get("predicted_label"):
            pred_label = pred.get("predicted_label")
        ply_to_gt[ply_name] = gt_label
        ply_to_preds[ply_name].append(pred_label)

    correct_ply = 0
    total_ply = 0
    no_pred_ply = 0
    confusion_ply = defaultdict(lambda: defaultdict(int))
    ply_results = []
    gt_score_ply: List[float] = []
    pred_score_ply: List[float] = []
    gt_mos_ply: List[float] = []
    pred_score_mos_ply: List[float] = []

    for ply, gt_label in ply_to_gt.items():
        votes = [p for p in ply_to_preds[ply] if p is not None]
        pred_ply = majority_vote(votes) if votes else None

        if pred_ply is None:
            no_pred_ply += 1
        else:
            gt_norm = gt_label if gt_label in LABELS else next((l for l in LABELS if l.lower() == gt_label.lower()), gt_label)
            if pred_ply.lower() == gt_norm.lower():
                correct_ply += 1
            confusion_ply[gt_norm][pred_ply] += 1
            gt_s = label_to_score(gt_label)
            pred_s = label_to_score(pred_ply)
            if gt_s is not None and pred_s is not None:
                gt_score_ply.append(gt_s)
                pred_score_ply.append(pred_s)
            mos = ply_to_mos.get(ply)
            if mos is not None and pred_s is not None:
                gt_mos_ply.append(mos)
                pred_score_mos_ply.append(pred_s)

        total_ply += 1
        ply_results.append({
            "ply_name": ply,
            "gt_label": gt_label,
            "pred_label": pred_ply,
            "votes": votes,
            "correct": pred_ply is not None and pred_ply.lower() == (gt_label or "").lower(),
        })

    acc_ply = (correct_ply / total_ply * 100) if total_ply else 0
    print("=" * 60)
    print("PCQA 5-label evaluation (per-point-cloud, majority vote over views)")
    print("=" * 60)
    print(f"Total point clouds (with GT): {total_ply}")
    print(f"Point clouds unparseable:     {no_pred_ply}")
    print(f"Accuracy:                     {correct_ply}/{total_ply} = {acc_ply:.2f}%")
    print()
    print("Confusion matrix (rows=GT, cols=Pred):")
    print("-" * 60)
    header = "            " + "  ".join(f"{p:>8}" for p in LABELS)
    print(header)
    for gt in LABELS:
        row = [confusion_ply[gt].get(p, 0) for p in LABELS]
        print(f"  {gt:10} " + "  ".join(f"{row[i]:>8}" for i in range(len(LABELS))))
    print()
    for label in LABELS:
        tp = confusion_ply[label].get(label, 0)
        pred_total = sum(confusion_ply[g].get(label, 0) for g in LABELS)
        gt_total = sum(confusion_ply[label].get(p, 0) for p in LABELS)
        prec = (tp / pred_total * 100) if pred_total else 0
        rec = (tp / gt_total * 100) if gt_total else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {label:10}  P: {prec:5.1f}%  R: {rec:5.1f}%  F1: {f1:5.1f}%  (n={gt_total})")
    if pearsonr is not None and spearmanr is not None and len(gt_score_ply) > 1:
        plcc, _ = pearsonr(gt_score_ply, pred_score_ply)
        srcc, _ = spearmanr(gt_score_ply, pred_score_ply)
        print()
        print("Correlation (per-point-cloud, GT=label→1-5, Pred=majority→1-5):")
        print(f"  PLCC: {plcc:.4f}  SRCC: {srcc:.4f}")
        if len(gt_mos_ply) > 1:
            plcc_mos, _ = pearsonr(gt_mos_ply, pred_score_mos_ply)
            srcc_mos, _ = spearmanr(gt_mos_ply, pred_score_mos_ply)
            print("  (GT=MOS 0-100, Pred=1-5):")
            print(f"  PLCC(MOS): {plcc_mos:.4f}  SRCC(MOS): {srcc_mos:.4f}")
    elif len(gt_score_ply) > 1:
        print("\nInstall scipy for PLCC/SRCC: pip install scipy")
    if args.out_ply_json:
        with open(args.out_ply_json, "w", encoding="utf-8") as f:
            json.dump(ply_results, f, indent=2, ensure_ascii=False)
        print(f"\nPer-ply results saved to {args.out_ply_json}")


if __name__ == "__main__":
    main()
