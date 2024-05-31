#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_models_large_multi_decoder.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/whisper_large_multi_log/decoder_all_der_5/slurm_log_%a.txt
#SBATCH --error=/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/whisper_large_multi_log/decoder_all_der_5/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --array=0-31
#SBATCH --exclusive

# args=(5)
layer_num=()
# ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}
for ((i=0; i<32; i++)); do
    layer_num+=("model.decoder.layers.$i.final_layer_norm")
done

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
        --function-name '${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --n-derangements 5 \
        --asr-option 'all' \
        --num-neurons 1280 \
        --save-plot-location '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/whisper_large_multi/decoder_all_der_5/plot/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --save-expression-set-location '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/whisper_large_multi/decoder_all_der_5/expression_set/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
  "

# cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/