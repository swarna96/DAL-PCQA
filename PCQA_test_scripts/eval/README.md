# Evaluation (PCQA Test Scripts)

Scripts to evaluate model predictions from LLaVA/InternVL inference.

**Inputs (common):**

- **Predictions JSON** — List of `{"id": "...", "text": "..."}` (e.g. `../answers/llava_zero_shot_pcqa_test.json`).
- **Test JSON** — LLaVA-format test set with `id` and either `image_A` or `image` (e.g. `../llava_data/test_pcqa_llava.json`).
- **Annotations** — CSV or XLSX file with `Ply_name` and optionally `Label`, `MOS`, or `Generated_Description` (column names configurable). For XLSX: `pip install pandas openpyxl`.

Paths below are relative to **`PCQA_test_scripts`** when running from that folder.

---

## 1. `eval_pcqa_labels.py` — 5-label accuracy & MOS correlation (per-point-cloud)

Evaluates predictions against **quality labels** (Excellent / Good / Fair / Poor / Bad) and optional MOS. **Per-point-cloud only:** majority vote over the four projections per ply, then accuracy and correlation over plies.

**Requirements:** `scipy` (for Pearson/Spearman).

```bash
cd PCQA_test_scripts
python eval/eval_pcqa_labels.py \
  --predictions answers/llava_zero_shot_pcqa_test.json \
  --test_json llava_data/test_pcqa_llava.json \
  --annotations /path/to/descriptions.csv
# Or Excel: --annotations /path/to/descriptions.xlsx
```

**Options:**

- `--ply_col` — CSV column for ply name (default: `Ply_name`).
- `--label_col` — CSV column for 5-level label (default: `Label`).
- `--mos_col` — CSV column for MOS (optional; enables correlation metrics).
- `--out_ply_json` — Optional: save per-ply pred/GT to JSON.

---

## 2. `eval_pcqa_descriptions.py` — Text vs reference (BLEU, ROUGE, BERTScore) (per-point-cloud)

Evaluates **model produced descriptions** against a reference text column (e.g. `Generated_Description`). **Per-point-cloud only:** average scores over the four views per ply, then mean over plies.

**Requirements:** `rouge-score`, `nltk` (and `python -c "import nltk; nltk.download('punkt')"` once). Optional: `bert-score`.

```bash
python eval/eval_pcqa_descriptions.py \
  --predictions answers/<answer_file> \
  --test_json llava_data/test_pcqa_llava.json \
  --annotations </path/to/annotation file> \
  --ref_col Generated_Description
```

**Options:**

- `--ply_col` — Ply name column (default: `Ply_name`).
- `--ref_col` — Reference text column (default: `Generated_Description`).
- `--bert_score` — Compute BERTScore (slower).
- `--out_ply_json` — Optional: save per-ply average scores to JSON.

---

## 3. `llm_judge_pcqa.py` — LLM-as-a-judge vs reference (1–5 score)

Uses a Hugging Face chat model (e.g. Llama 3.1 8B Instruct) to score how well each generated description **aligns with the reference** (1–5). Reference is required.

**Requirements:** `transformers`, `torch`, `accelerate`. For Llama: accept license and `huggingface-cli login`.

```bash
python eval/llm_judge_pcqa.py \
  --predictions answers/<answers_file> \
  --test_json llava_data/test_pcqa_llava.json \
  --annotations /path/to/annotations.csv \
  --ref_col Generated_Description \
  --out_json answers/llm_judge_scores.json
# Annotations can be .csv or .xlsx (for XLSX: pip install pandas openpyxl).
```

**Options:**

- `--model` — Hugging Face model (default: `meta-llama/Llama-3.1-8B-Instruct`).
- `--load_in_4bit` — Reduce VRAM (needs `bitsandbytes`).
- `--max_samples N` — Run on first N samples (for debugging).

**Shell script:**

```bash
./eval/run_llm_judge.sh [PREDICTIONS] [OUT_JSON] [TEST_JSON] ANNOTATIONS
```

Example: `./eval/run_llm_judge.sh answers/llava_zero_shot_pcqa_test.json answers/llm_judge_scores.json llava_data/test_pcqa_llava.json /path/to/descriptions.csv`. Run from `PCQA_test_scripts`.
