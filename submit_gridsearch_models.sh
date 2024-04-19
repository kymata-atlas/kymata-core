#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_models.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-toolbox-data/output/whisper_log/encoder_all_der_5/slurm_log_%a.txt
#SBATCH --error=kymata-toolbox-data/output/whisper_log/encoder_all_der_5/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=240G
#SBATCH --array=0-7
#SBATCH --exclusive

# args=(5)
layer_num=("model.encoder.conv1" "model.encoder.conv2" "model.encoder.layers.0.final_layer_norm" "model.encoder.layers.1.final_layer_norm" "model.encoder.layers.2.final_layer_norm" "model.encoder.layers.3.final_layer_norm" "model.encoder.layers.4.final_layer_norm" "model.encoder.layers.5.final_layer_norm")
# ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}

module load apptainer
apptainer exec \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c \
    " cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/ ; \
      export VENV_PATH=~/poetry/ ; \
      export VIRTUAL_ENV=/imaging/woolgar/projects/Tianyi/virtualenvs/kymata-toolbox-jvBImMG9-py3.11/ ; \
      \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
        --base-dir '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/' \
        --function-path 'predicted_function_contours/asr_models/whisper_all_no_reshape' \
        --function-name '${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --n-derangements 5 \
        --asr-option 'all' \
  "
