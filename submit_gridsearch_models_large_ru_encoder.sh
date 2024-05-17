#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_models_large_ru_encoder.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/whisper_large_encoder_log/slurm_log_%a.txt
#SBATCH --error=/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/whisper_large_encoder_log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --array=0-33
#SBATCH --exclusive

# args=(5)
layer_num=("model.encoder.conv1" "model.encoder.conv2")
# ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}
for ((i=0; i<31; i++)); do
    layer_num+=("model.encoder.layers.$i.final_layer_norm")
done

# # Printing layer_num variable to log file
# echo "layer_num variable: ${layer_num[@]}" >> "/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/whisper_large_ru_log/decoder_all_der_5/slurm_log_$SLURM_ARRAY_TASK_ID.txt"

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
        --base-dir '/imaging/projects/cbu/kymata/data/dataset_3-russian_narratives' \
        --function-path '/imaging/woolgar/projects/Tianyi/data/predicted_function_contours/asr_models/ru_whisper_all_no_reshape_large' \
        --function-name '${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --n-derangements 5 \
        --asr-option 'all' \
        --num-neurons 1280 \
        --save-plot-location '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/whisper_large_encoder/plot/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --save-expression-set-location '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/whisper_large_encoder/expression_set/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
  "

# cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/