#!/bin/bash

echo English

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/sensor/encoder/meg_sig_neurons_layer32_neuron1280.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/sensor/encoder"

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/sensor/decoder_text/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/sensor/decoder_text"

echo Russian

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text"

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/encoder/meg_sig_neurons_layer32_neuron1280.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/encoder"

echo English-Russian

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/encoder/meg_sig_neurons_layer32_neuron1280.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/encoder"

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text"
