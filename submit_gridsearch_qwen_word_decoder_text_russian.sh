#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/qwen_russian/sensor/decoder_text/log/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/qwen_russian/sensor/decoder_text/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=10G
#SBATCH --array=0-28

layer_num=()
for ((i=0; i<29; i++)); do
    layer_num+=("layer$i")
done

ORIG_HOME="$HOME"
export PATH="$ORIG_HOME/.local/bin:$PATH"

export XDG_CONFIG_HOME="${TMPDIR:-/tmp}/xdg_config_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HOME="${TMPDIR:-/tmp}/home_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

cd /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/

# Activate the project venv directly (avoids relying on Poetry being importable on compute nodes)
if [ -f "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate" ]; then
  source "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate"
else
  echo "ERROR: expected venv not found at .venv/bin/activate" 1>&2
  echo "Create it once on a login node (e.g. run 'poetry install') and re-submit." 1>&2
  exit 1
fi

# Now run your grid search with all layers and neurons
# /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/python kymata/invokers/run_gridsearch.py \
python kymata/invokers/run_gridsearch.py \
  --config dataset3.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/dataset_3-russian_narratives/predicted_function_contours/audio/asr_models/qwen2.5-omni-7b/decoder_text' \
  --transform-name "${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --n-derangements 5 \
  --asr-option 'all' \
  --num-neurons 3584 \
  --mfa True \
  --n-splits 400 \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --overwrite

  # --low-level-function \