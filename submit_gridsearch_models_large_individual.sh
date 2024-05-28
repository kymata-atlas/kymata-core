#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_models_large_individual.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/individual_log/slurm_log_encoder_%a.txt
#SBATCH --error=/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/individual_log/slurm_log_encoder_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --array=8-20
#SBATCH --exclusive

# args=(5)
# participant=("pilot_01" "pilot_02" "participant_01" "participant_01b" "participant_02" "participant_03" "participant_04" "participant_05" "participant_07" "participant_08" "participant_09" "participant_10" "participant_11" "participant_12" "participant_13" "participant_14" "participant_15" "participant_16" "participant_17" "participant_18" "participant_19")
participant=("participant_07" "participant_08" "participant_09" "participant_10" "participant_11" "participant_12" "participant_13" "participant_14" "participant_15" "participant_16" "participant_17" "participant_18" "participant_19")
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
        --base-dir '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/' \
        --function-path '/imaging/woolgar/projects/Tianyi/data/predicted_function_contours/asr_models/whisper_all_no_reshape_large_multi' \
        --function-name 'model.encoder.layers.10.final_layer_norm' \
        --n-derangements 5 \
        --asr-option 'all' \
        --num-neurons 1280 \
        --single-participant-override '${participant[$(($SLURM_ARRAY_TASK_ID - 8))]}' \
        --save-plot-location '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/individual/encoder/plot/${participant[$(($SLURM_ARRAY_TASK_ID - 8))]}' \
        --save-expression-set-location '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/individual/encoder/expression_set/${participant[$(($SLURM_ARRAY_TASK_ID - 8))]}' \
  "

# cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/
# --function-name 'model.decoder.layers.15.final_layer_norm' \