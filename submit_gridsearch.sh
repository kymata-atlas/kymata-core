#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=logs/slurm_log-%x.%j.out.txt
#SBATCH --error=logs/slurm_log-%x.%j.err.txt
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=240G
#SBATCH --array=1-1
#SBATCH --exclusive

args=(5) # 2 3 4 5 6 7 8 9 10)
ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}

module load apptainer
apptainer exec \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c \
    " cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/ ; \
      export VENV_PATH=~/poetry/ ; \
      export VIRTUAL_ENV=/imaging/woolgar/projects/Tianyi/virtualenvs/kymata-toolbox-jvBImMG9-py3.11/ ; \
      \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
        --config dataset4.yaml \
        --function-path 'predicted_function_contours/GMSloudness/stimulisig' \
        --function-name IL STL \
        --overwrite
  "
  #  --snr $ARG # >> result3.txt
        # --single-participant-override 'participant_01' \
        # --inverse-operator-dir '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/intrim_preprocessing_files/4_hexel_current_reconstruction/inverse-operators/' \
        # --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-megonly-emptyroom1-inv.fif'
  
  #  --function-path '/imaging/projects/cbu/kymata/data/open-source/ERP_CORE/MMN All Data and Scripts/functions/kymata_mr' \