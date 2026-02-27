#!/bin/bash
in="/imaging/projects/cbu/kymata/analyses/cai/kymata-core-tianyi-temp/kymata-core-data/tianyi/sensor/logs_split/"
out="/imaging/projects/cbu/kymata/analyses/cai/kymata-core-tianyi-temp/kymata-core-data/tianyi/sensor/scatter_split/"

declare -a modalities=("eeg" "meg")
declare -a axes=("latency" "neuron")

for modality in "${modalities[@]}"; do
  for axis in "${axes[@]}"; do
    poetry run python -m kymata.invokers.neuron_scatter \
      --dataset "${modality}" --x-axis "${axis}"\
      -i "${in}" -o "${out}"
  done
done
