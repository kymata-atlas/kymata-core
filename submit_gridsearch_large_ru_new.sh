#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_large_ru_new.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/paper/ru/log/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/paper/ru/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --array=0-63
#SBATCH --exclusive

# args=(5)
layer_num=()
# ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}
for ((i=0; i<32; i++)); do
    layer_num+=("model.encoder.layers.$i.fc2")
done

for ((i=0; i<32; i++)); do
    layer_num+=("model.decoder.layers.$i.fc2")
done

module load apptainer
apptainer exec \
  -B /imaging/woolgar/projects/Tianyi/ \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c \
    " cd /imaging/woolgar/projects/Tianyi/kymata-core/ ; \
      export VENV_PATH=~/poetry/ ; \
      export VIRTUAL_ENV=/imaging/woolgar/projects/Tianyi/virtualenvs/kymata-toolbox-jvBImMG9-py3.11/ ; \
      \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
        --config dataset4.yaml \
        --input-stream auditory \
        --plot-top-channels \
        --function-path '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output/temp/ru/whisper_fc2_and_final_layer_norm/whisper_large_teacher' \
        --num-neurons 1280 \
        --function-name '${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --n-derangements 5 \
        --asr-option 'all' \
        --mfa True \
        --save-plot-location '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output/paper/ru/plot/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
        --save-expression-set-location '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output/paper/ru/expression_set/${layer_num[$(($SLURM_ARRAY_TASK_ID))]}' \
  "
