#!/bin/bash
in="/Users/cai/Dox/Work/Kymata lab/Code/kymata-core/kymata-core-data/tianyi/sensor/logs_split/"
out="/Users/cai/Dox/Work/Kymata lab/Code/kymata-core/kymata-core-data/tianyi/sensor/scatter_split/"

declare -a modalities=("eeg" "meg")
declare -a axes=("latency" "neuron")

for modality in "${modalities[@]}"; do
  for axis in "${axes[@]}"; do
    poetry run python -m kymata.invokers.neuron_scatter \
      --dataset "${modality}" --x-axis "${axis}"\
      -i "${in}" -o "${out}"
  done
done
