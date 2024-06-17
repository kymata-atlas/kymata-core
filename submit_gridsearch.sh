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
    " cd /imaging/projects/cbu/kymata/analyses/andy/kymata-core/ ; \
      export VENV_PATH=~/poetry/ ; \
      \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
        --config dataset4.yaml \
        --input-stream auditory \
        --function-path 'predicted_function_contours/GMSloudness/stimulisig' \
        --function-name IL STL IL1 IL2 IL3 IL4 IL5 IL6 IL7 IL8 IL9  \
        --overwrite
  "
  #  --snr $ARG # >> result3.txt
