#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch.sh
###

#SBATCH --job-name=gridsearch
#SBATCH --output=logs/slurm_log-%x.%j.out.txt
#SBATCH --error=logs/slurm_log-%x.%j.trace.txt
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=240G
#SBATCH --array=1-1
#SBATCH --exclusive

args=(5) # 2 3 4 5 6 7 8 9 10)
ARG=${args[$SLURM_ARRAY_TASK_ID - 1]}

export PATH="$HOME/.local/bin:$PATH"
# Change this to your own path to where kymata is installed.
# If running from kymata-core, this is the path to kymata-core.
# If running with kymata as a dependency, it is the root directory of your project.
cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/
source $(poetry env info --path)/bin/activate
python -m kymata.invokers.run_gridsearch \
  --config dataset4.yaml \
  --input-stream auditory \
  --transform-path 'predicted_function_contours/GMSloudness/stimulisig' \
  --transform-name IL STL IL1 IL2 IL3 IL4 IL5 IL6 IL7 IL8 IL9  \
  --plot-top-channels \
  --overwrite

#  --snr $ARG # >> result3.txt
