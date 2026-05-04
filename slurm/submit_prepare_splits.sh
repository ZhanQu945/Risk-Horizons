#!/usr/bin/env bash
#SBATCH --job-name=grail_splits
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-python}
SEEDS=${SEEDS:-13,17,23,29,31}

# Provide dataset-specific processed inputs.
MIMIC_DATA_PATH=${MIMIC_DATA_PATH:-./Processed/mimiciv/ehr_tables_full_withlab.pkl}
MIMIC_LABEL_MAP=${MIMIC_LABEL_MAP:-./Processed/mimiciv/ehr_label_map.json}
MIMIC_LAB_GROUPS=${MIMIC_LAB_GROUPS:-./Processed/mimiciv/ehr_lab_groups.csv}
EICU_DATA_PATH=${EICU_DATA_PATH:-./Processed/eicu/ehr_tables_full_withlab.pkl}
EICU_LABEL_MAP=${EICU_LABEL_MAP:-./Processed/eicu/ehr_label_map.json}
EICU_LAB_GROUPS=${EICU_LAB_GROUPS:-./Processed/eicu/ehr_lab_groups.csv}

${PYTHON_BIN} run_prepare_splits.py \
  --mimic-data-path "${MIMIC_DATA_PATH}" \
  --mimic-label-map "${MIMIC_LABEL_MAP}" \
  --mimic-lab-groups "${MIMIC_LAB_GROUPS}" \
  --eicu-data-path "${EICU_DATA_PATH}" \
  --eicu-label-map "${EICU_LABEL_MAP}" \
  --eicu-lab-groups "${EICU_LAB_GROUPS}" \
  --seeds "${SEEDS}" \
  --min-visits 2 \
  --train-ratio 0.8 \
  --val-ratio 0.1
