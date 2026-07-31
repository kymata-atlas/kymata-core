#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/qwen_ecog_high_freq/log/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/qwen_ecog_high_freq/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=10G
#SBATCH --array=0-28
#SBATCH --exclusive

NET_TMP="/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_ecog_high_freq/tmp/task_${SLURM_ARRAY_TASK_ID}"
export XDG_CONFIG_HOME="${NET_TMP}/xdg_config"
export XDG_CACHE_HOME="${NET_TMP}/xdg_cache"
export MPLCONFIGDIR="${NET_TMP}/mpl"
export NUMBA_CACHE_DIR="${NET_TMP}/numba"
export TMPDIR="${NET_TMP}/general"

mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TMPDIR"

cd /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/

# 3. Environment activation check from your template
if [ -f "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate" ]; then
  source "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate"
else
  echo "ERROR: expected venv not found at .venv/bin/activate" 1>&2
  echo "Create it once on a login node (e.g. run 'poetry install') and re-submit." 1>&2
  exit 1
fi

layer_num=()
for ((i=0; i<29; i++)); do
    layer_num+=("layer$i")
done

python kymata/invokers/run_gridsearch.py \
  --config dataset_ecog_mean.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --emeg-t-start 0 \
  --transform-path '/imaging/projects/cbu/kymata/data/open-source/ECoG/predicted_function_contours/asr_models/qwen/decoder_text' \
  --transform-name "${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --n-derangements 5 \
  --asr-option 'all' \
  --num-neurons 3584 \
  --mfa True \
  --n-splits 1798 \
  --single-participant-override 'sub-kmeans300-norm' \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_ecog_high_freq/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_ecog_high_freq/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --freq-band 'high' \
  --overwrite

rm -rf "$NET_TMP"
