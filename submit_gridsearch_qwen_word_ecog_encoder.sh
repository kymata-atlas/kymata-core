#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/ecog_language/qwen_encoder/log/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/ecog_language/qwen_encoder/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=10G
#SBATCH --array=0-31

layer_num=()
for ((i=0; i<32; i++)); do
    layer_num+=("layer$i")
done

export PATH="$HOME/.local/bin:$PATH"

export XDG_CONFIG_HOME="${TMPDIR:-/tmp}/xdg_config_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"

cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/
source $(poetry env info --path)/bin/activate

# Now run your grid search with all layers and neurons
python kymata/invokers/run_gridsearch.py \
  --config dataset_ecog_mean.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --emeg-t-start 0 \
  --transform-path '/imaging/projects/cbu/kymata/data/open-source/ECoG/predicted_function_contours/asr_models/qwen/encoder' \
  --transform-name "${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --n-derangements 5 \
  --asr-option 'all' \
  --num-neurons 1280 \
  --mfa True \
  --n-splits 1798 \
  --single-participant-override 'sub-kmeans300' \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ecog_language/qwen_encoder/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ecog_language/qwen_encoder/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --overwrite

  # --low-level-function \