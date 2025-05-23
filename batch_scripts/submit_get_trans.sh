#!/bin/bash

###
# To run get_feats on the queue at the CBU, run the following command in command line:
#   sbatch submit_get_trans.sh
###


#SBATCH --job-name=get_trans
#SBATCH --output=slurm_log_trans.txt
#SBATCH --error=slurm_log_trans.txt
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=240G
#SBATCH --array=0-4
#SBATCH --exclusive

size=('small' 'base' 'large' 'large-v2' 'medium')

module load apptainer
apptainer exec \
  -B /imaging/woolgar/projects/Tianyi/ \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c \
    " cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/ ; \
      export VENV_PATH=~/poetry/ ; \
      export HF_HOME=/imaging/woolgar/projects/Tianyi/models ; \
      export VIRTUAL_ENV=/imaging/woolgar/projects/Tianyi/virtualenvs/kymata-toolbox-jvBImMG9-py3.11/ ; \
      \$VENV_PATH/bin/poetry run python -m get_trans \
        --size '${size[$(($SLURM_ARRAY_TASK_ID))]}' \
  "

# -B /imaging/projects/cbu/kymata/ \