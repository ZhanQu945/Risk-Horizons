#!/usr/bin/env bash
#SBATCH --job-name=grail_main
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-python}
RUN_NAME=${RUN_NAME:-grail_main}
SEED=${SEED:-13}

DATA_PATH=${DATA_PATH:-./Processed/mimiciv/ehr_tables_full_withlab.pkl}
LABEL_MAP=${LABEL_MAP:-./Processed/mimiciv/ehr_label_map.json}
LAB_GROUPS=${LAB_GROUPS:-./Processed/mimiciv/ehr_lab_groups.csv}
SPLIT_FILE=${SPLIT_FILE:-./Processed/mimiciv/splits/ehr_tables_full_withlab_seed13_tr0.80_val0.10_mv2.json}

${PYTHON_BIN} -u run_experiment.py \
  --data-path "${DATA_PATH}" \
  --format pickle \
  --label-map "${LABEL_MAP}" \
  --lab-groups "${LAB_GROUPS}" \
  --text-mode text \
  --disable-lab-tests \
  --min-visits 2 \
  --min-code-freq 0 \
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
  --early-stop-patience 5 \
  --early-stop-min-delta 0.0003 \
  --early-stop-max-traj 600 \
  --early-stop-metric recall@10 \
  --max-local-candidates 16000 \
  --max-candidates 100 \
  --cone-aperture 0.80 \
  --seed "${SEED}" \
  --split-file "${SPLIT_FILE}" \
  --save-all-samples \
  --run-name "${RUN_NAME}"
