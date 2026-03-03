# PCQA Test Scripts (LLaVA & InternVL)

This folder contains scripts to run **zero-shot** and **fine-tuned** LLaVA and InternVL on the PCQA (Point Cloud Quality Assessment) dataset.

## Layout

- **`llava/`** — LLaVA zero-shot inference, fine-tuning, and data conversion.
- **`internvl/`** — InternVL zero-shot inference and fine-tuning.
- **`depictqa/`** — DepictQA model: config, train/infer scripts, and JSON generator (calls repo `src/`; no code duplication).
- **`dataset/`** — Place your projections here (or set `DATASET_ROOT`; see below).
- **`llava_data/`** — Output of the LLaVA-format conversion; shell scripts expect it here or set `LLAVA_DATA`.
- **`answers/`** — Inference outputs (predictions JSON) by default.

## Dataset

Put the PCQA dataset under **`dataset/`** so that:

- `dataset/<projections_folder>/` contains projection images (paths like `<projections_folder>/<ply_name>/<view_id>.png`).

Alternatively, set **`DATASET_ROOT`** to the absolute path of your dataset. Generate LLaVA-format train/test JSON from CSV or XLSX using the converter in `llava/` (see below).

## LLaVA

1. **Convert PCQA to LLaVA format** — from a **Annotation file** and a **test split file** (list of test ply names):

   ```bash
   cd llava
   python convert_pcqa_to_llava_format.py \
     --descriptions_csv path/to/<annotation_file> \
     --test_split_csv path/to/sjtu_data_info/<test_split_csv> \
     --root_dir ../dataset \
     --out_dir ../llava_data \
     --projections_subdir projections \
     --ply_column Ply_name \
     --description_column Generated_Description \
     --test_ply_column name
   ```
   
   - If your files are Excel (`.xlsx`) instead of CSV, use `--descriptions_xlsx` and `--test_split_xlsx` with the paths to those files. The converter writes `train_pcqa_llava.json` and `test_pcqa_llava.json` under `--out_dir`.

2. **Zero-shot inference:**
   ```bash
   cd llava
   ./run_llava_zero_shot.sh [GPU_ID]
   ```
   Uses `../llava_data/test_pcqa_llava.json` and `../dataset` by default; override with `LLAVA_DATA` and `DATASET_ROOT` if needed.

3. **Fine-tune:**
   ```bash
   cd llava
   ./finetune_llava.sh
   ```

4. **Inference with finetuned (LoRA) model:**
   ```bash
   cd llava
   ./infer_llava_finetuned.sh [ADAPTER_DIR] [OUTPUT_JSON]
   ```

## InternVL


Use **-hf** models only (e.g. `OpenGVLab/InternVL3-8B-hf`). InternVL2-8B (non-hf) is not supported by these scripts.

1. **Zero-shot inference:**
   ```bash
   cd internvl
   ./run_internvl_zero_shot.sh [TEST_JSON] [ROOT_DIR] [OUTPUT] [MODEL_NAME]
   ```
   Defaults: `../llava_data/test_pcqa_llava.json`, `../dataset`, `../answers/internvl_zero_shot_pcqa_test.json`, `OpenGVLab/InternVL3-8B-hf`.

2. **Fine-tune:**
   ```bash
   cd internvl
   ./finetune_internvl.sh [MODEL_NAME] [OUTPUT_DIR]
   ```

3. **Inference with finetuned model:**
   ```bash
   cd internvl
   ./infer_internvl_finetuned.sh [ADAPTER_DIR] [OUTPUT_JSON]
   ```

## DepictQA (train & infer)

Scripts and config in **`depictqa/`** train and run inference with the DepictQA model on PCQA. They call the repo’s `src/train.py` and `src/infer.py`. 
1. **Generate train/test JSON** (DepictQA format) from annotations CSV and projections:
   ```bash
   cd depictqa
   python gen_pcqa_depictqa_json.py path/to/annotations.csv path/to/projections_dir path/to/out.json --test-split-csv path/to/<test_split_file>
   ```
   See `depictqa/README.md` for options (`--ply-col`, `--generated-col`, etc.).

2. **Edit `depictqa/config.yaml`** — Set `data.root_dir` (projections + JSON), `data.train.meta_paths_weights`, and model paths (vision encoder, LLM, optional `delta_path` checkpoint).

3. **Train:** `cd depictqa && ./train.sh <gpu_id>` (DeepSpeed).

4. **Infer:** `cd depictqa && ./infer_pcqa.sh <gpu_id> [test_meta_path] [dataset_name]`. Output goes to `config.infer.answer_dir`; use it with `eval/` scripts below.

See **`depictqa/README.md`** for full steps and paths.

## Evaluation

Predictions are written as JSON: a list of `{"id": "...", "text": "..."}`. Use the scripts in **`eval/`**:

- **`eval/eval_pcqa_labels.py`** — 5-label accuracy and MOS correlation (needs annotations CSV with `Ply_name`, `Label`; optional `MOS`).
- **`eval/eval_pcqa_descriptions.py`** — BLEU, ROUGE (and optional BERTScore) vs a reference description column (e.g. `Generated_Description`).
- **`eval/llm_judge_pcqa.py`** — LLM-as-a-judge 1–5 score (with or without reference). Run via **`eval/run_llm_judge.sh [predictions] [out_json]`** from this folder.

See **`eval/README.md`** for usage, options, and paths (test JSON: `llava_data/test_pcqa_llava.json`, annotations CSV with `Ply_name` and label/description columns).
