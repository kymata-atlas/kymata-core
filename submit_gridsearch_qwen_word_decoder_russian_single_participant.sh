#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/qwen_russian/sensor/decoder_text/meg15_0082/log/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/qwen_russian/sensor/decoder_text/meg15_0082/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=50G
#SBATCH --array=0-0


export PATH="$HOME/.local/bin:$PATH"

cd /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/

npy_file='/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text/dec_neurons.npy'
    
layers=()
neurons=()

source $(poetry env info --path)/bin/activate
# source /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate

echo "Starting to read the npy file"

for i in {0..423}; do
  output=$(python kymata/invokers/read_npy.py $npy_file $i)

  a=$(echo $output | awk '{printf "%d", $1}')
  b=$(echo $output | awk '{printf "%d", $2}')

  layers+=("layer${a}")
  neurons+=("${b}")
done

# Convert arrays to space-separated strings
layers_string=$(printf "%s " "${layers[@]}")
neurons_string=$(printf "%s " "${neurons[@]}")

echo "Number of elements in neurons: ${#neurons[@]}"
echo "Starting doing gridsearch now"

# Now run your grid search with all layers and neurons
# /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/python kymata/invokers/run_gridsearch.py \
python kymata/invokers/run_gridsearch.py \
  --config dataset3.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/dataset_3-russian_narratives/predicted_function_contours/audio/asr_models/qwen2.5-omni-7b/decoder_text' \
  --num-neurons $neurons_string \
  --transform-name $layers_string \
  --n-derangements 5 \
  --asr-option 'some' \
  --mfa True \
  --n-splits 400 \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text/meg15_0082/expression" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text/meg15_0082/expression" \
  --overwrite \
  --single-participant-override 'meg15_0082' \

  # --low-level-function \