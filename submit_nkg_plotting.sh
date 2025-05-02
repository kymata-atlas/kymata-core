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

cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/
poetry run python kymata/invokers/invoker_run_nkg_plotting.py