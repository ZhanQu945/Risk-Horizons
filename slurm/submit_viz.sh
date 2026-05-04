#!/usr/bin/env bash
#SBATCH --job-name=grail_viz
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-python}
GEOM_ARTIFACT=${GEOM_ARTIFACT:?Set GEOM_ARTIFACT to a geometry_artifacts.pt path}
RUN_NAME=${RUN_NAME:-viz_run}

${PYTHON_BIN} run_export_viz_inputs.py \
  --artifact "${GEOM_ARTIFACT}" \
  --run-name "${RUN_NAME}"

# Edit VIZ_INPUT_DIR to the output directory from run_export_viz_inputs.py
VIZ_INPUT_DIR=${VIZ_INPUT_DIR:?Set VIZ_INPUT_DIR to exported visualization input directory}

${PYTHON_BIN} hyperbolic_viz.py \
  --input-dir "${VIZ_INPUT_DIR}"
