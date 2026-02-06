#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/qwen_english_russian/sensor/decoder_text/log/slurm_log_15.txt
#SBATCH --error=kymata-core-data/output/qwen_english_russian/sensor/decoder_text/log/slurm_log_15.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=10G
#SBATCH --array=0-0
#SBATCH --exclusive

layer_num=()
for ((i=0; i<29; i++)); do
    layer_num+=("layer$i")
done

export PATH="$HOME/.local/bin:$PATH"

cd /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/
source $(poetry env info --path)/bin/activate
# source /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate

# Now run your grid search with all layers and neurons
# /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/python kymata/invokers/run_gridsearch.py \
python kymata/invokers/run_gridsearch.py \
  --config dataset4.1.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/dataset_4.1-russian_narrative_english_native/predicted_function_contours/asr_models/qwen2.5-omni-7b/decoder_text' \
  --transform-name "layer15" \
  --n-derangements 5 \
  --asr-option 'all' \
  --num-neurons 3584 \
  --mfa True \
  --n-splits 400 \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text/expression/layer15" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text/expression/layer15" \
  --overwrite

  # --low-level-function \