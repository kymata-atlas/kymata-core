#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_models.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-toolbox-data/output/whisper_log/decoder_k_der_5/slurm_log_%a.txt
#SBATCH --error=kymata-toolbox-data/output/whisper_log/decoder_k_der_5/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=240G
#SBATCH --array=0-5
#SBATCH --exclusive

# args=(5)
layer_num=("model.decoder.layers.0.encoder_attn.k_proj" "model.decoder.layers.1.encoder_attn.k_proj" "model.decoder.layers.2.encoder_attn.k_proj" "model.decoder.layers.3.encoder_attn.k_proj" "model.decoder.layers.4.encoder_attn.k_proj" "model.decoder.layers.5.encoder_attn.k_proj")
# ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}

module load apptainer
apptainer exec \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c \
    " cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/ ; \
      export VENV_PATH=~/poetry/ ; \
      \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
        --base-dir '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/' \
        --function-path 'predicted_function_contours/asr_models/whisper_all_no_reshape' \
        --function-name '${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --n-derangements 5 \
        --asr-option 'all' \
  "
