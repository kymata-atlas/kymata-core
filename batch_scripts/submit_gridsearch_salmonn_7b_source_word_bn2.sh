#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/first_speech_paper/all_word_source/slurm_log_batch_n2.txt
#SBATCH --error=kymata-core-data/output/first_speech_paper/all_word_source/slurm_log_batch_n2.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=200G
#SBATCH --array=0-0
#SBATCH --exclusive

export PATH="$HOME/.local/bin:$PATH"

cd /imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/
npy_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/new_batch/word.npy'
    
layers=()
neurons=()

source $(poetry env info --path)/bin/activate

echo "Starting to read the npy file"

for i in {82..163}; do
  output=$(python kymata/invokers/read_npy.py $npy_file $i)

  a=$(echo $output | awk '{print $1}')
  b=$(echo $output | awk '{print $2}')

  layers+=("layer${a}")
  neurons+=("${b}")
done

# Convert arrays to space-separated strings
layers_string=$(printf "%s " "${layers[@]}")
neurons_string=$(printf "%s " "${neurons[@]}")

echo "Number of elements in neurons: ${#neurons[@]}"
echo "Starting doing gridsearch now"

# Now run your grid search with all layers and neurons
python kymata/invokers/run_gridsearch.py \
  --config dataset4.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/predicted_function_contours/asr_models/salmonn/7B' \
  --num-neurons $neurons_string \
  --transform-name $layers_string \
  --n-derangements 5 \
  --asr-option 'some' \
  --mfa True \
  --save-plot-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_word_source/batch_n2' \
  --save-expression-set-location '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_word_source/batch_n2' \
  --use-inverse-operator \
  --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-fusion-inv.fif' \
  --morph
