#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch.sh
###

#SBATCH --job-name=gridsearch
#SBATCH --output=slurm_log_trialwise.txt
#SBATCH --error=slurm_log_trialwise_e.txt
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=1000
#SBATCH --array=1-1
#SBATCH --exclusive

conda activate mne_venv

python3 invokers/run_preprocessing_dataset4.py

conda deactivate
