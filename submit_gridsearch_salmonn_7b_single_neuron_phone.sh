#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_salmonn_7b_single_neuron_phone.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/single_neuron/phone/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/single_neuron/phone/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --array=0-0
#SBATCH --exclusive

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
        --input-stream auditory \
        --plot-top-channels \
        --function-path '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/predicted_function_contours/asr_models/salmonn_phone/7B' \
        --num-neurons 990 \
        --function-name 'layer10' \
        --n-derangements 5 \
        --asr-option 'one' \
        --mfa True \
        --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/single_neuron/phone' \
        --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/single_neuron/phone' \
  "

        # --use-inverse-operator \
        # --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-fusion-inv.fif' \
        # --morph \