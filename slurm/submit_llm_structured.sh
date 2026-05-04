#!/usr/bin/env bash
#SBATCH --job-name=grail_llm_struct
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
RUN_NAME=${RUN_NAME:-llm_structured}
SEED=${SEED:-13}

DATA_PATH=${DATA_PATH:-./Processed/mimiciv/ehr_tables_full_withlab.pkl}
LABEL_MAP=${LABEL_MAP:-./Processed/mimiciv/ehr_label_map.json}
LAB_GROUPS=${LAB_GROUPS:-./Processed/mimiciv/ehr_lab_groups.csv}
SPLIT_FILE=${SPLIT_FILE:-./Processed/mimiciv/splits/ehr_tables_full_withlab_seed13_tr0.80_val0.10_mv2.json}

LLM_BACKEND=${LLM_BACKEND:-hf}
LLM_MODEL=${LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TEXT_MODE=${TEXT_MODE:-text}

${PYTHON_BIN} -u scripts/llm/run_llm_only_freq_candidates.py \
  --data-path "${DATA_PATH}" \
  --format pickle \
  --label-map "${LABEL_MAP}" \
  --lab-groups "${LAB_GROUPS}" \
  --text-mode "${TEXT_MODE}" \
  --disable-lab-tests \
  --llm-backend "${LLM_BACKEND}" \
  --llm-model "${LLM_MODEL}" \
  --seed "${SEED}" \
  --split-file "${SPLIT_FILE}" \
  --save-all-samples \
  --run-name "${RUN_NAME}"
