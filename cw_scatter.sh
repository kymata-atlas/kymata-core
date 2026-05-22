#!/bin/bash
in="/Users/cai/Dox/Work/Kymata lab/Code/kymata-core/kymata-core-data/tianyi/sensor/encoder/log_split"
out="/Users/cai/Dox/Work/Kymata lab/Code/kymata-core/kymata-core-data/tianyi/sensor/encoder/scatter_split/"

declare -a modalities=("eeg" "meg")
declare -a axes=("latency" "neuron")

for modality in "${modalities[@]}"; do
  for axis in "${axes[@]}"; do
    poetry run python -m kymata.invokers.language_neuron_scatter \
      --dataset "${modality}" --x-axis "${axis}"\
      -i "${in}" -o "${out}"
  done
done
