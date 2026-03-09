#!/bin/bash
in="/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_word/expression_set"
out="/imaging/projects/cbu/kymata/analyses/cai/kymata-core-tianyi-temp/kymata-core-data/tianyi/sensor/expression_sets_split/"

for i in {0..32}; do
  poetry run python -m kymata.invokers.split_meg_eeg -o "${out}" -i "${in}/layer${i}/layer${i}_4095_gridsearch.nkg"
done
