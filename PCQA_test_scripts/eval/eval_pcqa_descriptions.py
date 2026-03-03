#!/usr/bin/env python3
"""
Evaluate generated quality descriptions (free-form text) against reference descriptions.

Metrics: BLEU, ROUGE-1/2/L (and optional BERTScore). Reference from annotations CSV/XLSX (e.g. Generated_Description).

Usage:
  pip install rouge-score nltk   # optional: bert-score; for XLSX: pandas openpyxl

  python eval_pcqa_descriptions.py \\
    --predictions ../answers/llava_zero_shot_pcqa_test.json \\
    --test_json ../llava_data/test_pcqa_llava.json \\
    --annotations path/to/generated_averaged_descriptions.csv \\
    --ref_col Generated_Description
  # Or Excel: --annotations path/to/descriptions.xlsx
"""

import argparse
import csv
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import nltk
    from nltk.translate import bleu_score as nltk_bleu
    NLTK_BLEU_AVAILABLE = True
except ImportError:
    NLTK_BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


def normalize_text(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip())


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower().strip()
    return s.split() if s else []


def get_bleu(hyp: str, ref: str) -> float:
    ref_tok = _tokenize(ref)
    hyp_tok = _tokenize(hyp)
    if not ref_tok or not hyp_tok:
        return 0.0
    if NLTK_BLEU_AVAILABLE:
        try:
            return nltk_bleu.sentence_bleu([ref_tok], hyp_tok)
        except Exception:
            pass
    ref_set = set(ref_tok)
    match = sum(1 for w in hyp_tok if w in ref_set)
    return match / len(hyp_tok) if hyp_tok else 0.0


def get_rouge(hyp: str, ref: str) -> Dict[str, float]:
    if not ROUGE_AVAILABLE:
        return {"rouge1": float("nan"), "rouge2": float("nan"), "rougeL": float("nan")}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(ref, hyp)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def _image_path_to_ply(image_path: str) -> str:
    parts = (image_path or "").replace("\\", "/").split("/")
    for p in parts:
        if p.endswith(".ply"):
            return p
    if "/" in image_path:
        return parts[0]
    if image_path.endswith(".png"):
        return image_path.rsplit(".", 2)[0]
    return image_path or ""


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
    parser = argparse.ArgumentParser(description="Evaluate generated quality descriptions (text)")
    parser.add_argument("--predictions", type=str, required=True, help="Inference output JSON (id, text)")
    parser.add_argument("--test_json", type=str, required=True, help="Test JSON (id, image_A or image)")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Annotations file: CSV or XLSX with Ply_name and reference text column")
    parser.add_argument("--ply_col", type=str, default="Ply_name")
    parser.add_argument("--ref_col", type=str, default="Generated_Description", help="CSV column for reference description")
    parser.add_argument("--bert_score", action="store_true", help="Compute BERTScore (slower, needs bert-score)")
    parser.add_argument("--out_ply_json", type=str, default=None, help="Optional: save per-ply average scores to JSON")
    args = parser.parse_args()

    with open(args.test_json, encoding="utf-8") as f:
        test_data = json.load(f)
    id_to_image = {}
    for item in test_data:
        pid = item.get("id")
        path = item.get("image_A") or item.get("image")
        if pid is not None and path:
            id_to_image[pid] = path

    ply_to_ref: Dict[str, str] = {}
    for row in _load_annotations_rows(args.annotations):
        ply = (row.get(args.ply_col) or "").strip()
        ref = (row.get(args.ref_col) or "").strip()
        if ply and ref:
            ply_to_ref[ply] = normalize_text(ref)

    with open(args.predictions, encoding="utf-8") as f:
        pred_data = json.load(f)

    pairs: List[Tuple[str, str, str, str]] = []
    for pred in pred_data:
        pid = pred.get("id")
        text = pred.get("text", "")
        if pid not in id_to_image:
            continue
        image_path = id_to_image[pid]
        ply_name = _image_path_to_ply(image_path)
        ref = ply_to_ref.get(ply_name)
        if ref is None:
            continue
        hyp = normalize_text(text)
        pairs.append((pid, ply_name, hyp, ref))

    if not pairs:
        print("No (prediction, reference) pairs found. Check test_json and annotations file.")
        return

    bleu_scores: List[float] = []
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougeL_scores: List[float] = []

    for _pid, _ply_name, hyp, ref in pairs:
        bleu = get_bleu(hyp, ref)
        rouge = get_rouge(hyp, ref)
        bleu_scores.append(bleu)
        rouge1_scores.append(rouge["rouge1"])
        rouge2_scores.append(rouge["rouge2"])
        rougeL_scores.append(rouge["rougeL"])

    bert_f1: List[float] = []
    if args.bert_score and BERTSCORE_AVAILABLE and pairs:
        hyps = [p[2] for p in pairs]
        refs = [p[3] for p in pairs]
        _P, _R, F1 = bert_score_fn(hyps, refs, lang="en", verbose=False)
        bert_f1 = F1.tolist()

    # Aggregate by ply (avg over views per ply, then mean over plies)
    ply_bleu: Dict[str, List[float]] = {}
    ply_r1: Dict[str, List[float]] = {}
    ply_r2: Dict[str, List[float]] = {}
    ply_rL: Dict[str, List[float]] = {}
    ply_bert: Dict[str, List[float]] = {}
    for i, (_, ply_name, _, _) in enumerate(pairs):
        ply_bleu.setdefault(ply_name, []).append(bleu_scores[i])
        ply_r1.setdefault(ply_name, []).append(rouge1_scores[i])
        ply_r2.setdefault(ply_name, []).append(rouge2_scores[i])
        ply_rL.setdefault(ply_name, []).append(rougeL_scores[i])
        if args.bert_score and BERTSCORE_AVAILABLE and i < len(bert_f1):
            ply_bert.setdefault(ply_name, []).append(bert_f1[i])
    n_ply = len(ply_bleu)
    if not n_ply:
        print("No point clouds with reference found.")
        return

    mean_bleu = sum(sum(s) / len(s) for s in ply_bleu.values()) / n_ply
    mean_r1 = sum(sum(s) / len(s) for s in ply_r1.values()) / n_ply
    mean_r2 = sum(sum(s) / len(s) for s in ply_r2.values()) / n_ply
    mean_rL = sum(sum(s) / len(s) for s in ply_rL.values()) / n_ply

    print("=" * 60)
    print("Quality description evaluation (per-point-cloud: avg over views, then mean over plies)")
    print("=" * 60)
    print(f"Point clouds (with reference): {n_ply}")
    print()
    print("Metric:")
    print(f"  BLEU:     {mean_bleu:.4f}")
    if ROUGE_AVAILABLE:
        print(f"  ROUGE-1:  {mean_r1:.4f}")
        print(f"  ROUGE-2:  {mean_r2:.4f}")
        print(f"  ROUGE-L:  {mean_rL:.4f}")
    else:
        print("  ROUGE:    (pip install rouge-score)")
    if args.bert_score and BERTSCORE_AVAILABLE and ply_bert:
        mean_bert = sum(sum(s) / len(s) for s in ply_bert.values()) / n_ply
        print(f"  BERTScore F1: {mean_bert:.4f}")
    elif args.bert_score and not BERTSCORE_AVAILABLE:
        print("  BERTScore: (pip install bert-score)")

    if args.out_ply_json:
        ply_means = []
        for ply in ply_bleu:
            rec = {"ply_name": ply, "bleu": sum(ply_bleu[ply]) / len(ply_bleu[ply])}
            if ROUGE_AVAILABLE:
                rec["rouge1"] = sum(ply_r1[ply]) / len(ply_r1[ply])
                rec["rouge2"] = sum(ply_r2[ply]) / len(ply_r2[ply])
                rec["rougeL"] = sum(ply_rL[ply]) / len(ply_rL[ply])
            if args.bert_score and BERTSCORE_AVAILABLE and ply in ply_bert:
                rec["bert_score_f1"] = sum(ply_bert[ply]) / len(ply_bert[ply])
            ply_means.append(rec)
        with open(args.out_ply_json, "w", encoding="utf-8") as f:
            json.dump(ply_means, f, indent=2, ensure_ascii=False)
        print(f"\nPer-ply average scores saved to {args.out_ply_json}")


if __name__ == "__main__":
    main()
