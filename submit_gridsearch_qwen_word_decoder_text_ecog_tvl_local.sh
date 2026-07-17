#!/bin/bash

npy_file='/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_ecog/decoder_text/analysis/sig_neurons.npy'
LOGFILE="/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_ecog/decoder_text/tvl/log/slurm_log_0.txt"
CHUNK_ID=0
CHUNK_COUNT=1

echo "Reading neuron selections from: ${npy_file}" | tee "$LOGFILE"

chunk_out=$(poetry run python kymata/invokers/read_npy.py "$npy_file" --chunk "$CHUNK_ID" "$CHUNK_COUNT" 2>&1)
if [ $? -ne 0 ]; then
  echo "ERROR: failed to read selections via read_npy.py" | tee -a "$LOGFILE"
  echo "$chunk_out" | tee -a "$LOGFILE"
  exit 2
fi

echo "$chunk_out" | tee -a "$LOGFILE"

layers_string=$(echo "$chunk_out" | awk -F= '/^layers=/{sub(/^layers=/,"",$0); print $0}')
neurons_string=$(echo "$chunk_out" | awk -F= '/^neurons=/{sub(/^neurons=/,"",$0); print $0}')

if [ -z "$neurons_string" ] || [ -z "$layers_string" ]; then
  echo "No neurons/layers in this chunk (local run). Exiting." | tee -a "$LOGFILE"
  exit 0
fi

num_neurons=$(echo "$neurons_string" | wc -w)
echo "Number of neurons in this chunk: ${num_neurons}" | tee -a "$LOGFILE"
echo "Starting gridsearch now" | tee -a "$LOGFILE"

poetry run python kymata/invokers/run_gridsearch.py \
  --config dataset_ecog.yaml \
  --input-stream auditory \
  --plot-top-channels \
  --transform-path '/imaging/projects/cbu/kymata/data/open-source/ECoG/predicted_function_contours/asr_models/qwen/decoder_text' \
  --num-neurons $neurons_string \
  --transform-name $layers_string \
  --n-derangements 5 \
  --asr-option 'some' \
  --mfa True \
  --low-level-function \
  --n-splits 1798 \
  --save-plot-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_ecog/decoder_text/tvl/expression" \
  --save-expression-set-location "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_ecog/decoder_text/tvl/expression" \
  --start-latency -0.5 \
  --overwrite 2>&1 | tee -a "$LOGFILE"