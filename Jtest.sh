#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch.sh
###

#SBATCH --job-name=gridsearch
#SBATCH --output=logs/slurm_log-%x.%j.out.txt
#SBATCH --error=logs/slurm_log-%x.%j.trace.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=240G
#SBATCH --array=0-0
#SBATCH --exclusive

# part_num=()
# for ((i=1; i<22; i++)); do
#     part_num+=("$i")
# done

export PATH="$HOME/.local/bin:$PATH"
source $(poetry env info --path)/bin/activate
python kymata/invokers/run_gridsearch.py \
  --config dataset4.yaml \
  --input-stream auditory \
  --transform-path 'predicted_function_contours/GMSloudness/stimulisig' \
  --transform-name IL STL IL1 IL2 IL3 IL4 IL5 IL6 IL7 IL8 IL9  \
  --plot-top-channels \
  --emeg-dir 'interim_preprocessing_files/3_trialwise_sensorspace/evoked_data_last_four_reps/' \
  --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/last_four_reps_tvl' \
  --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/last_four_reps_tvl' \
  --use-inverse-operator \
  --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-fusion-inv.fif' \
  --morph \
  --overwrite

#  --snr $ARG # >> result3.txt
