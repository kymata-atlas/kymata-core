#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_models_large_individual_ru.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/meg15_0045/log/slurm_log_%a.txt
#SBATCH --error=/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/meg15_0045/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --array=0-7
#SBATCH --exclusive

# args=(5)
layer_num=("model.encoder.conv1" "model.encoder.conv2" "model.encoder.layers.10.final_layer_norm" "model.encoder.layers.20.final_layer_norm" "model.encoder.layers.30.final_layer_norm" "model.decoder.layers.10.final_layer_norm" "model.decoder.layers.20.final_layer_norm" "model.decoder.layers.30.final_layer_norm")
# participant=("participant_07" "participant_08" "participant_09" "participant_10" "participant_11" "participant_12" "participant_13" "participant_14" "participant_15" "participant_16" "participant_17" "participant_18" "participant_19")
# ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}

module load apptainer
apptainer exec \
  -B /imaging/woolgar/projects/Tianyi/ \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c \
    " cd /imaging/woolgar/projects/Tianyi/kymata-toolbox/ ; \
      export VENV_PATH=~/poetry/ ; \
      export VIRTUAL_ENV=/imaging/woolgar/projects/Tianyi/virtualenvs/kymata-toolbox-jvBImMG9-py3.11/ ; \
      \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
        --config 'dataset3.yaml' \
        --function-path '/imaging/woolgar/projects/Tianyi/data/predicted_function_contours/asr_models/ru_whisper_all_no_reshape_large' \
        --function-name '${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --n-derangements 5 \
        --asr-option 'all' \
        --num-neurons 1280 \
        --single-participant-override 'meg15_0045' \
        --save-plot-location '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/meg15_0045/plot/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --save-expression-set-location '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/meg15_0045/expression_set/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
  "

# cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/
# --function-name model.decoder.layers.15.final_layer_norm \