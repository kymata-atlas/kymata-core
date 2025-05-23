#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/first_speech_paper/all_morpheme_source/slurm_log_batch_2.txt
#SBATCH --error=kymata-core-data/output/first_speech_paper/all_morpheme_source/slurm_log_batch_2.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=200G
#SBATCH --array=0-0
#SBATCH --exclusive

module load apptainer

apptainer exec \
  -B /imaging/woolgar/projects/Tianyi/ \
  -B /imaging/projects/cbu/kymata/ \
  /imaging/local/software/singularity_images/python/python_3.11.7-slim.sif \
  bash -c "
    cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/ ;
    export VENV_PATH=~/poetry/ ;
    export VIRTUAL_ENV=/imaging/woolgar/projects/Tianyi/virtualenvs/kymata-toolbox-jvBImMG9-py3.11/ ;
    npy_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/morpheme_all.npy' ;
    
    layers=() ;
    neurons=() ;
    
    for i in {176..352}; do
      output=\$(\$VENV_PATH/bin/poetry run python -m kymata.invokers.read_npy \$npy_file \$i) ;

      a=\$(echo \$output | awk '{print \$1}') ;
      b=\$(echo \$output | awk '{print \$2}') ;

      layers+=(\"layer\${a}\") ;
      neurons+=(\"\${b}\") ;
    done

    # Convert arrays to space-separated strings
    layers_string=\$(printf \"%s \" \"\${layers[@]}\") ;
    neurons_string=\$(printf \"%s \" \"\${neurons[@]}\") ;

    # Now run your grid search with all layers and neurons
    \$VENV_PATH/bin/poetry run python -m kymata.invokers.run_gridsearch \
      --config dataset4.yaml \
      --input-stream auditory \
      --plot-top-channels \
      --transform-path '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/predicted_function_contours/asr_models/salmonn_morpheme/7B' \
      --num-neurons \$neurons_string \
      --transform-name \$layers_string \
      --n-derangements 5 \
      --asr-option 'some' \
      --mfa True \
      --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_morpheme_source/batch_2' \
      --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_morpheme_source/batch_2' \
      --use-inverse-operator \
      --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-fusion-inv.fif' \
      --morph
  "
