#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/qwen_russian/sensor/encoder/log/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/qwen_russian/sensor/encoder/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=10G
#SBATCH --array=0-31
#SBATCH --exclusive

layer_num=()
for ((i=0; i<32; i++)); do
    layer_num+=("layer$i")
done

export PATH="$HOME/.local/bin:$PATH"

cd /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/
source $(poetry env info --path)/bin/activate
# source /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate

# Now run your grid search with all layers and neurons
# /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/python kymata/invokers/run_gridsearch.py \
python kymata/invokers/run_gridsearch.py \
  --config dataset3.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/dataset_3-russian_narratives/predicted_function_contours/audio/asr_models/qwen2.5-omni-7b/encoder' \
  --transform-name "${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --n-derangements 5 \
  --asr-option 'all' \
  --num-neurons 1280 \
  --mfa True \
  --n-splits 411 \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/encoder/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/encoder/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --overwrite

  # --low-level-function \