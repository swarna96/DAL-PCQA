# DepictQA (PCQA train & infer)

Scripts and config to **train** and **run inference** with the DepictQA model on PCQA. They call the DepictQA repo’s `src/train.py` and `src/infer.py`; 

**Only for DepictQA:** The steps below (clone DepictQA, set environment, train, infer) apply **only when you want to run the DepictQA model**. LLaVA, InternVL, and the eval scripts in `../eval/` have their own environments and do not require cloning DepictQA or this setup.

---

## 0. Clone DepictQA and set up environment (only for DepictQA train/infer)

1. **Clone the DepictQA repository** and go to the repo root:
   ```bash
   git clone <depictqa-repo-url> DepictQA-main
   cd DepictQA-main
   ```

2. **Create and activate the environment** (from the DepictQA root) as per DepictQA’s instructions (e.g. `conda env create -f environment.yml` and `conda activate dmProject`, or `pip install -r requirements.txt`).

3. **Set the repo root.** Before running `train.sh` or `infer_pcqa.sh`, set `DEPICTQA_ROOT` to the path of the DepictQA repo root (the directory that contains `src/`):
   ```bash
   export DEPICTQA_ROOT=/path/to/DepictQA-main
   ```
   
4. **Train and test:** from this folder run in order: generate train/test JSON (step 1) → edit config (step 2) → train (step 3) → infer (step 4). Steps 1–4 are below.

---

## 1. Generate train/test JSON (DepictQA format)

From annotations CSV and projection images:

```bash
cd depictqa
python gen_pcqa_depictqa_json.py \
  path/to/annotations.csv \
  path/to/projections_dir \
  path/to/output_dir/out.json \
  --test-split-csv path/to/<test_split_file>\
  --test-csv-name-col name \
  --ply-col Ply_name \
  --generated-col Generated_Description
```

- **Without** `--test-split-csv`: writes a single JSON to `output_dir/out.json`.
- **With** `--test-split-csv`: writes under `output_dir/`:
  - `train_pcqa.json`
  - `test_pcqa.json` (inference format: id, image_A, query, etc.)

Paths in the JSON are relative to `projections_dir`. Use the same path as `config.data.root_dir` when training, and put the generated JSON files inside that directory (or set paths in config to a subdir).

---

## 2. Config

Edit **`config.yaml`**:

- **`data.root_dir`** — Directory that contains projection images and the train/test JSON files (e.g. `../dataset` or an absolute path). Paths inside the JSON are relative to this.
- **`data.train.meta_paths_weights`** — List of `[filename, weight]`; filename is under `root_dir`. If you used `gen_pcqa_depictqa_json.py --test-split-csv`, set this to `[[train_pcqa.json, 1]]` unless you renamed the file.
- **`model.*`** — Vision encoder, LLM, and checkpoint paths. In the template these are relative to the DepictQA repo root (e.g. `ModelZoo/CLIP/clip/ViT-L-14.pt`). Set **`model.delta_path`** to a pretrained DepictQA checkpoint  or leave `null` to train from scratch.

---

## 3. Train

Set `DEPICTQA_ROOT` (see step 0.3), then:

```bash
cd depictqa
export DEPICTQA_ROOT=/path/to/DepictQA-main   # if not already set
./train.sh <gpu_id>
```

This runs `deepspeed` and DepictQA’s `src/train.py` with `config.yaml` from this folder. Checkpoints and logs go to the paths in `config.yaml` (e.g. `./ckpt/`, `./log/` relative to DepictQA root unless you change them).

---

## 4. Infer

Ensure the test JSON (e.g. `test_pcqa.json`) is under `config.data.root_dir` or that the path you pass is resolvable. Set `DEPICTQA_ROOT` (see step 0.3), then:

```bash
cd depictqa
export DEPICTQA_ROOT=/path/to/DepictQA-main   # if not already set
./infer_pcqa.sh <gpu_id> [test_meta_path] [dataset_name]
```

Example (defaults: `test_pcqa.json`, `pcqa_baseline`):

```bash
./infer_pcqa.sh 0
```

Example with explicit test file and output name:

```bash
./infer_pcqa.sh 0 test_pcqa.json pcqa_test
```

Output is written under `config.infer.answer_dir` (e.g. `./answers/`) as `quality_single_A_noref_<dataset_name>.json`. Use that file with the eval scripts in `../eval/` (labels, descriptions, LLM judge).

---

## 5. Optional: fixed prompt

To use a single prompt for all test samples, create e.g. `structured_prompt_pcqa.txt` in this folder and in `infer_pcqa.sh` uncomment the line that passes `--prompt "$(cat ...)"` to `infer.py`.
