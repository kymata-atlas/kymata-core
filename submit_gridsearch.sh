#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=slurm_log.txt
#SBATCH --error=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=1000
#SBATCH --array=1-1
#SBATCH --exclusive

conda activate mne_venv

args=(5) # 2 3 4 5 6 7 8 9 10)
ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}

python invokers/run_gridsearch.py \
  --base-dir "/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/" \
  --data-path "intrim_preprocessing_files/3_trialwise_sensorspace/evoked_data" \
  --function-path "predicted_function_contours/GMSloudness/stimulisig" \
  --function-name "d_IL2" \
  --emeg-file "participant_01-ave"
#  --snr $ARG # >> result3.txt

conda deactivate
