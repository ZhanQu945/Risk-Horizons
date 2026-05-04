# GRAIL: Geometry-Aware Retrieval-Augmented Inference with LLMs over Hyperbolic Representations of Patient Trajectories

This repository contains the PyTorch implementation of GRAIL for longitudinal EHR next-visit prediction, including:
- hyperbolic geometry model (main method),
- LLM inference variants (structured candidate scoring with expansion),
- baseline models,
- preprocessing and split generation,
- visualization and evaluation utilities.

No run outputs are included; minimal de-identified Slurm job scripts are provided in `slurm/` for reproducibility.

## 1) Repository Layout

- `scripts/`: entrypoint scripts grouped by function
  - `preprocess/`, `experiments/`, `llm/`, `analysis/`, `viz/`
- `grail/`: core library
  - preprocessing utilities, graph construction, model, retrieval, training, evaluation, baselines
- `scripts/experiments/run_experiment.py`: main end-to-end runner (geometry + optional LLM reranking + baselines)
- `scripts/preprocess/preprocess_mimic_iv.py`, `scripts/preprocess/preprocess_eicu.py`: dataset preprocessing
- `scripts/experiments/run_prepare_splits.py`, `scripts/experiments/run_prepare_splits_eicu.py`: fixed split generation/reuse
- `scripts/llm/run_llm_only_freq_candidates.py`: LLM-only structured-candidate inference with expansion/retrieval policies
- `scripts/experiments/run_pretrained_ehr_baseline.py`: pretrained BEHRT/MedBERT baseline launcher
- `scripts/viz/hyperbolic_viz.py`, `scripts/viz/run_export_viz_inputs.py`, `scripts/viz/run_visualize_artifact.py`: geometry visualization tools
- `scripts/analysis/run_seed_summary.py`: multi-seed aggregation
- `scripts/analysis/run_analyze_failure_modes.py`: post-hoc failure mode analysis

## 2) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Data Format

Input table is visit-level with at least:
- `subject_id`
- visit timestamp (`admittime` by default)
- modality columns (defaults):
  - `icd_code` (dx)
  - `pro_code` (proc)
  - `ndc` (med)
  - `lab_test` (lab; optional)

Each modality column should contain a list of codes per visit.

## 4) Preprocessing

MIMIC-IV:
```bash
python scripts/preprocess/preprocess_mimic_iv.py --help
```

eICU:
```bash
python scripts/preprocess/preprocess_eicu.py --help
```

## 5) Split Generation / Reuse

General split prep (multiple seeds, both datasets):
```bash
python scripts/experiments/run_prepare_splits.py --help
```

eICU-only split prep helper:
```bash
python scripts/experiments/run_prepare_splits_eicu.py --help
```

All experiments support `--split-file` so train/val/test can be fixed and reused.

## 6) Main Experiment (Strong Hyperbolic GRAIL)

Example command template:
```bash
python -u scripts/experiments/run_experiment.py \
  --data-path <processed_table.pkl> \
  --format pickle \
  --label-map <label_map.json> \
  --lab-groups <lab_groups.csv> \
  --text-mode text \
  --min-visits 2 \
  --min-code-freq 0 \
  --disable-lab-tests \
  --baseline grail \
  --llm-backend none \
  --pipeline-stage geometry \
  --barycenter-heads 8 \
  --barycenter-mixture \
  --multi-cone-retrieval \
  --include-intra-modality-pmi \
  --require-joint-lag01 \
  --joint-lag01-min-support 3 \
  --pmi-thresh 0.05 \
  --persistence-bonus 0.8 \
  --seed 13 \
  --split-file <split.json> \
  --early-stop-patience 5 \
  --early-stop-min-delta 0.0003 \
  --early-stop-max-traj 600 \
  --early-stop-metric recall@10 \
  --max-local-candidates 16000 \
  --max-candidates 100 \
  --cone-aperture 0.80 \
  --save-all-samples \
  --run-name <run_name>
```

## 7) LLM Variants

### 7.1 Geometry + LLM reranking
Use `scripts/experiments/run_experiment.py` with:
- `--pipeline-stage llm`
- `--geometry-artifact-path <geometry_artifacts.pt>`
- `--llm-backend {hf|openai|heuristic|ftcls}`

This path uses structured candidates with optional expansion and scores candidates at inference time.

### 7.2 LLM-only structured candidates (+ expansion/policies)
```bash
python scripts/llm/run_llm_only_freq_candidates.py --help
```

### 7.3 Fine-tuned LLM classifier
Train:
```bash
python scripts/experiments/run_llm_finetune.py --help
```
Use as reranker via `scripts/experiments/run_experiment.py --llm-backend ftcls --ft-llm-checkpoint <ckpt>`.

## 8) Baselines

Run through `scripts/experiments/run_experiment.py --baseline <name>`:
- `transformer`
- `retain`
- `gram`
- `behrt`
- `medbert`
- `halo`
- `copy_last`
- `hyper_no_stitch`
- `euclid_grail`
- `euclid_rag`

Pretrained BEHRT/MedBERT helper:
```bash
python scripts/experiments/run_pretrained_ehr_baseline.py --help
```

## 9) Major Ablations (Flags)

Ablations are controlled in `scripts/experiments/run_experiment.py`, including:
- geometry manifold (`--manifold`),
- no-stitch/euclidean variants (baseline switches),
- multi-head / mixture / multi-cone,
- PMI/support controls (`--pmi-thresh`, `--joint-lag01-min-support`, intra-modality PMI flags),
- retrieval size controls (`--max-local-candidates`, `--max-candidates`),
- persistence prior (`--persistence-bonus`),
- LLM weighting (`--lambda-llm`).

## 10) Visualization

Export visualization inputs from artifacts:
```bash
python scripts/viz/run_export_viz_inputs.py --help
```

Generate hyperbolic embedding figures:
```bash
python scripts/viz/hyperbolic_viz.py --help
```

Artifact-level visualization:
```bash
python scripts/viz/run_visualize_artifact.py --help
```

## 11) Evaluation Outputs

Typical run output directory contains:
- metrics JSON (`*_geometry_metrics.json` or `*_all_metrics.json`),
- runtime summary JSON,
- split stats JSON,
- sample outputs JSON (full test predictions if `--save-all-samples`).

Implemented metrics include:
- Recall@K, Precision@K, nDCG@K, MRR
- macro aggregates
- persistent vs novel buckets
- seen-train vs unseen-train buckets (and crossed buckets with novelty)
- hierarchy-aware metrics (where applicable)
- efficiency metrics

## 12) Slurm Job Scripts (Minimal)

Each Slurm script is a thin wrapper around one Python entrypoint:
- `slurm/submit_prepare_splits.sh` -> `scripts/experiments/run_prepare_splits.py`
- `slurm/submit_grail_main.sh` -> `scripts/experiments/run_experiment.py` (main GRAIL config)
- `slurm/submit_llm_structured.sh` -> `scripts/llm/run_llm_only_freq_candidates.py`
- `slurm/submit_baseline.sh` -> `scripts/experiments/run_experiment.py --baseline <name>`
- `slurm/submit_viz.sh` -> `scripts/viz/run_export_viz_inputs.py` then `scripts/viz/hyperbolic_viz.py`

These scripts are de-identified and path-agnostic; configure inputs/models through environment variables.

Examples:
```bash
sbatch slurm/submit_prepare_splits.sh
sbatch slurm/submit_grail_main.sh
BASELINE=transformer sbatch slurm/submit_baseline.sh
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct sbatch slurm/submit_llm_structured.sh
GEOM_ARTIFACT=/path/to/geometry_artifacts.pt VIZ_INPUT_DIR=/path/to/viz_inputs sbatch slurm/submit_viz.sh
```
