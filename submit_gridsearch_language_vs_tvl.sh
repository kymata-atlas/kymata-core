#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_language_vs_tvl.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/low_level_func/log/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/low_level_func/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --array=0-65
#SBATCH --exclusive

# args=(5)
layer_num=("model.encoder.conv1" "model.encoder.conv2")
# ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}
for ((i=0; i<32; i++)); do
    layer_num+=("model.encoder.layers.$i.final_layer_norm")
done
for ((i=0; i<32; i++)); do
    layer_num+=("model.decoder.layers.$i.final_layer_norm")
done

module load apptainer
apptainer exec \
  -B /imaging/woolgar/projects/Tianyi/ \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c \
    " cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/ ; \
      export VENV_PATH=~/poetry/ ; \
      export VIRTUAL_ENV=/imaging/woolgar/projects/Tianyi/virtualenvs/kymata-toolbox-jvBImMG9-py3.11/ ; \
      \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
        --config dataset4.yaml \
        --plot-top-channels \
        --input-stream auditory \
        --function-path '/imaging/woolgar/projects/Tianyi/data/predicted_function_contours/asr_models/whisper_all_no_reshape_large_multi' \
        --function-name '${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --n-derangements 5 \
        --asr-option 'all' \
        --num-neurons 1280 \
        --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/low_level_func/expression_set/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/low_level_func/plot/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --overwrite \
        --low-level-function \
        --start-latency -500 \
  "
