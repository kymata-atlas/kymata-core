#!/bin/bash
in="/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/expression"
out="/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/nkg_split"

for i in {0..28}; do
  poetry run python -m kymata.invokers.split_meg_eeg -o "${out}" -i "${in}/layer${i}/layer${i}_3583_gridsearch.nkg"
done
