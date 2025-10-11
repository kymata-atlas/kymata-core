#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/salmonn_omni_listen_tf/sensor/log/slurm_log_sensor_%a.txt
#SBATCH --error=kymata-core-data/output/salmonn_omni_listen_tf/sensor/log/slurm_log_sensor_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=10G
#SBATCH --array=0-32
#SBATCH --exclusive

layer_num=()
for ((i=0; i<33; i++)); do
    layer_num+=("layer_$i")
done

export PATH="$HOME/.local/bin:$PATH"

cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/
source $(poetry env info --path)/bin/activate

# Now run your grid search with all layers and neurons
python kymata/invokers/run_gridsearch.py \
  --config dataset4.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/predicted_function_contours/asr_models/salmonn_omni/listen_tf' \
  --transform-name "${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --n-derangements 5 \
  --asr-option 'all' \
  --num-neurons 4096 \
  --mfa True \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/salmonn_omni_listen_tf/sensor/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/salmonn_omni_listen_tf/sensor/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --overwrite

  # --low-level-function \