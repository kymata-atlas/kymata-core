#!/bin/bash

###
# Not for running directly, called through the original run_gridsearch code wehn parallelising
###

#SBATCH --job-name=sub_grid
#SBATCH --output=slurm_log.txt
#SBATCH --error=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=1000
#SBATCH --array=1-$1
#SBATCH --exclusive

module load apptainer
apptainer exec \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c \
    ' cd /imaging/projects/cbu/kymata/analyses/andy/kymata-toolbox/ ; \
      export VENV_PATH=~/poetry/ ; \
      $VENV_PATH/bin/poetry run python invokers/run_gridsearch.py \
        --save-dict-path $2 \
        --result-dict-path {$3}_{$SLURM_ARRAY_TASK_ID}.npz \
        --proc-num $SLURM_ARRAY_TASK_ID
  '
