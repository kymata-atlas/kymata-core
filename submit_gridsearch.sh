#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch.sh
###

#SBATCH --job-name=gridsearch
#SBATCH --output=logs/slurm_log-%x.%j.out.txt
#SBATCH --error=logs/slurm_log-%x.%j.trace.txt
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=240G
#SBATCH --array=1-1
#SBATCH --exclusive

args=(5) # 2 3 4 5 6 7 8 9 10)
ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}

export PATH="$HOME/.local/bin:$PATH"
cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/ # Change to your own path to kymata-core
source $(poetry env info --path)/bin/activate
python kymata/invokers/run_gridsearch.py \
  --config dataset5.yaml \
  --input-stream tactile \
  --transform-path 'predicted_function_contours/stimulisig_new' \
  --transform-name LHSquareVib LHslowfluct RHSquareVib RHslowfluct  \
  --plot-top-channels \
  --emeg-dir 'interim_preprocessing_files/3_trialwise_sensorspace/evoked_data' \
  --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/tactile/new_fwd/two_reps_without_11_thumb' \
  --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/tactile/new_fwd/two_reps_without_11_thumb' \
  --overwrite \
  --use-inverse-operator \
  --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-megonly-emptyroom60-inv.fif' \
  --morph \
  --n-derangements 6 \
  --n-splits 400