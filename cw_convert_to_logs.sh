#!/bin/bash
in="/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/nkg_split/"
out="/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/log_split/"

declare -a modalities=("eeg" "meg")

for i in {0..28}; do
  for emeg in "${modalities[@]}"; do
    poetry run python -m kymata.invokers.invoker_exp2log \
      -o "${out}" -i "${in}/layer${i}_3583_gridsearch_${emeg}.nkg"
  done
done
