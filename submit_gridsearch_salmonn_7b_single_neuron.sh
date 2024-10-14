#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch_salmonn_7b_single_neuron.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/paper/single_neuron/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/paper/single_neuron/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --array=0-74
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
      npy_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/word_sig.npy' ; \
      output=\$(\$VENV_PATH/bin/poetry run python -m invokers.read_npy \$npy_file \$((SLURM_ARRAY_TASK_ID))) ; \

      a=\$(echo \$output | awk '{print \$1}') ; \
      b=\$(echo \$output | awk '{print \$2}') ; \

      echo \"Layer: \$a\" ; \
      echo \"Neuron: \$b\" ; \
      
      # Use the variables a and b here for further processing
      \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
        --config dataset4.yaml \
        --input-stream auditory \
        --plot-top-channels \
        --function-path '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/predicted_function_contours/asr_models/salmonn/7B' \
        --num-neurons \$b \
        --function-name "layer\${a}" \
        --n-derangements 5 \
        --asr-option 'one' \
        --mfa True \
        --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron/\${a}_\${b}" \
        --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron/\${a}_\${b}" \
        --use-inverse-operator \
        --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-fusion-inv.fif' \
        --morph \


      # \$VENV_PATH/bin/poetry run python -m invokers.run_gridsearch \
      #   --config dataset4.yaml \
      #   --input-stream auditory \
      #   --plot-top-channels \
      #   --function-path '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/predicted_function_contours/asr_models/salmonn/7B' \
      #   --num-neurons 2298 \
      #   --function-name 'layer6' \
      #   --n-derangements 5 \
      #   --asr-option 'one' \
      #   --mfa True \
      #   --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron' \
      #   --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron' \
      #   --use-inverse-operator \
      #   --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-fusion-inv.fif' \
      #   --morph \
  "
