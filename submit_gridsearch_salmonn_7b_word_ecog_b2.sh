#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/ecog_language/single_electrode/log_b2/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/ecog_language/single_electrode/log_b2/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=200G
#SBATCH --array=32-32
#SBATCH --exclusive

layer_num=()
for ((i=0; i<33; i++)); do
    layer_num+=("layer$i")
done

export PATH="$HOME/.local/bin:$PATH"

cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/
source $(poetry env info --path)/bin/activate

# Now run your grid search with all layers and neurons
python kymata/invokers/run_gridsearch.py \
  --config dataset_ecog.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/open-source/ECoG/predicted_function_contours/asr_models/salmonn/7B' \
  --transform-name "${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --n-derangements 5 \
  --asr-option 'all' \
  --num-neurons 2048 \
  --batch 'last' \
  --mfa True \
  --n-splits 1798 \
  --single-participant-override 'sub-all' \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ecog_language/single_electrode/expression_b2/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ecog_language/single_electrode/expression_b2/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --overwrite

  # --low-level-function \