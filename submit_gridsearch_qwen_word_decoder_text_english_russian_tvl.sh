#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/tvl/log/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/tvl/log/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=10G
#SBATCH --array=0-0

# layer_num=()
# for ((i=0; i<29; i++)); do
#     layer_num+=("layer$i")
# done

ORIG_HOME="$HOME"
export PATH="$ORIG_HOME/.local/bin:$PATH"

export XDG_CONFIG_HOME="${TMPDIR:-/tmp}/xdg_config_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HOME="${TMPDIR:-/tmp}/home_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

cd /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/
# source $(poetry env info --path)/bin/activate
# source /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate

# Now run your grid search with all layers and neurons
# /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/python kymata/invokers/run_gridsearch.py \
# python kymata/invokers/run_gridsearch.py \

if [ -f "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate" ]; then
  source "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/.venv/bin/activate"
else
  echo "ERROR: expected venv not found at .venv/bin/activate" 1>&2
  echo "Create it once on a login node (e.g. run 'poetry install') and re-submit." 1>&2
  exit 1
fi

npy_file='/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/sig_neurons.npy'
    
layers=()
neurons=()

echo "Starting to read the npy file"

for i in {0..113}; do
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

python kymata/invokers/run_gridsearch.py \
  --config dataset4.1.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --emeg-dir 'interim_preprocessing_files/3_trialwise_sensorspace/evoked_data/eng_rus/' \
  --transform-path '/imaging/projects/cbu/kymata/data/dataset_4.1-russian_narrative_english_native/predicted_function_contours/asr_models/qwen2.5-omni-7b/decoder_text' \
  --num-neurons $neurons_string \
  --transform-name $layers_string \
  --n-derangements 5 \
  --asr-option 'some' \
  --mfa True \
  --low-level-function \
  --n-splits 400 \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/tvl/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/tvl/expression/${layer_num[$SLURM_ARRAY_TASK_ID]}" \
  --start-latency -0.5 \
  --overwrite

  # --low-level-function \