#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/qwen_meg_high_freq/log/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/qwen_meg_high_freq/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=10G
#SBATCH --array=0-28
#SBATCH --exclusive

NET_TMP="/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_meg_high_freq/tmp/task_${SLURM_ARRAY_TASK_ID}"
export XDG_CONFIG_HOME="${NET_TMP}/xdg_config"
export XDG_CACHE_HOME="${NET_TMP}/xdg_cache"
export MPLCONFIGDIR="${NET_TMP}/mpl"
export NUMBA_CACHE_DIR="${NET_TMP}/numba"
export TMPDIR="${NET_TMP}/general"

mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TMPDIR"

VENV_PYTHON="/home/cw04/.cache/pypoetry/virtualenvs/kymata-YfOhN41B-py3.11/bin/python"

cd /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/

layer_num=()
for ((i=0; i<29; i++)); do
    layer_num+=("layer$i")
done

"$VENV_PYTHON" kymata/invokers/run_gridsearch.py \
  --config dataset4.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/predicted_function_contours/asr_models/qwen/decoder_text' \
  --transform-name "${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --n-derangements 5 \
  --asr-option 'all' \
  --num-neurons 3584 \
  --mfa True \
  --n-splits 400 \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_meg_high_freq/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_meg_high_freq/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --freq-band 'high' \
  --overwrite

rm -rf "$NET_TMP"
