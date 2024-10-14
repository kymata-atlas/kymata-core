#!/bin/bash

###
# To run get_feats on the queue at the CBU, run the following command in command line:
#   sbatch submit_nkg_plotting.sh
###


#SBATCH --job-name=get_feats
#SBATCH --output=slurm_log.txt
#SBATCH --error=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=240G
#SBATCH --array=1-1
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
      \$VENV_PATH/bin/poetry run python -m invokers.invoker_run_nkg_plotting \
  "

# -B /imaging/projects/cbu/kymata/ \