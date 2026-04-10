#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --output=kymata-core-data/output/qwen_english/source/log/cai/slurm_log_%a.txt
#SBATCH --error=kymata-core-data/output/qwen_english/source/log/cai/slurm_log_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=200G
#SBATCH --array=0-8
#SBATCH --exclusive

ORIG_HOME="$HOME"
export PATH="$ORIG_HOME/.local/bin:$PATH"

export XDG_CONFIG_HOME="${TMPDIR:-/tmp}/xdg_config_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HOME="${TMPDIR:-/tmp}/home_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

VENV_PYTHON="/home/cw04/.cache/pypoetry/virtualenvs/kymata-YfOhN41B-py3.11/bin/python"

cd /imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/

npy_file='/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/sensor/decoder_text/analysis/sig_neurons.npy'
    
layers=()
neurons=()

echo "Starting to read the npy file"

for i in {1934..2901}; do
  output=$("$VENV_PYTHON" kymata/invokers/read_npy.py $npy_file $i)

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

# Split work into 9 array tasks as evenly as possible.
total=${#neurons[@]}
n_jobs=9
chunk=$(( (total + n_jobs - 1) / n_jobs ))
start=$(( SLURM_ARRAY_TASK_ID * chunk ))
end=$(( start + chunk ))
if [ $end -gt $total ]; then
  end=$total
fi
if [ $start -ge $total ]; then
  echo "Nothing to do for task ${SLURM_ARRAY_TASK_ID} (start=${start} >= total=${total})."
  exit 0
fi

sel_layers=("${layers[@]:$start:$((end-start))}")
sel_neurons=("${neurons[@]:$start:$((end-start))}")
layers_string=$(printf "%s " "${sel_layers[@]}")
neurons_string=$(printf "%s " "${sel_neurons[@]}")

echo "Task ${SLURM_ARRAY_TASK_ID}: processing indices [${start}, ${end}) => n=$((end-start))"

"$VENV_PYTHON" kymata/invokers/run_gridsearch.py \
  --config dataset4.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/predicted_function_contours/asr_models/qwen/decoder_text' \
  --num-neurons $neurons_string \
  --transform-name $layers_string \
  --n-derangements 5 \
  --asr-option 'some' \
  --mfa True \
  --n-splits 400 \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/source/expression/task_${SLURM_ARRAY_TASK_ID}" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/source/expression/task_${SLURM_ARRAY_TASK_ID}" \
  --use-inverse-operator \
  --inverse-operator-suffix '_ico5-3L-loose02-cps-nodepth-fusion-inv.fif' \
  --morph

  # --low-level-function \