#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=logs/slurm_log-%x.%j.out.txt
#SBATCH --error=logs/slurm_log-%x.%j.trace.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=240G
#SBATCH --array=0-0
#SBATCH --exclusive

# part_num=()
# for ((i=1; i<22; i++)); do
#     part_num+=("$i")
# done

module load apptainer
apptainer exec \
  -B /imaging/woolgar/projects/Tianyi/ \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c \
    " cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core ; \
      export VENV_PATH=~/poetry/ ; \
      export VIRTUAL_ENV=/imaging/woolgar/projects/Tianyi/virtualenvs/kymata-toolbox-jvBImMG9-py3.11/
      \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
        --config dataset4.yaml \
        --input-stream auditory \
        --function-path 'predicted_function_contours/GMSloudness/stimulisig' \
        --use-inverse-operator \
        --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-fusion-inv.fif' \
        --function-name IL STL IL1 IL2 IL3 IL4 IL5 IL6 IL7 IL8 IL9  \
        --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output' \
        --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output' \
        --save-name 'test_chao' \
        --overwrite \
        --morph \
  "
  #  --snr $ARG # >> result3.txt
        # --single-participant-override 'participant_01' \
        # --inverse-operator-dir '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/intrim_preprocessing_files/4_hexel_current_reconstruction/inverse-operators/' \
        # --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-megonly-emptyroom1-inv.fif'
  
  #  --function-path '/imaging/projects/cbu/kymata/data/open-source/ERP_CORE/MMN All Data and Scripts/functions/kymata_mr' \

      # export HF_HOME=/imaging/woolgar/projects/Tianyi/models ; \
      # export VIRTUAL_ENV=/imaging/woolgar/projects/Tianyi/virtualenvs/kymata-toolbox-jvBImMG9-py3.11/ ; \