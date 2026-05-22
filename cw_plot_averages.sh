#!/bin/bash

echo English Encoder

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/sensor/encoder/meg_sig_neurons_layer32_neuron1280.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/english_encoder/scatter_split/"

echo English Decoder

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english/sensor/decoder_text/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/english_decoder/scatter_split/"

echo Russian Encoder

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/encoder/meg_sig_neurons_layer32_neuron1280.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/russian_encoder/scatter_split/"
poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/encoder/eeg_sig_neurons_layer32_neuron1280.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/russian_encoder/scatter_split/"
poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/encoder/meg_sig_neurons_layer32_neuron1280.npy" \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/encoder/eeg_sig_neurons_layer32_neuron1280.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/russian_encoder/scatter_split/"

echo Russian Decoder

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_russian/sensor/decoder_text/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/russian_decoder/scatter_split/"

echo English/Russian Encoder

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
   -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/encoder/meg_sig_neurons_layer32_neuron1280.npy" \
   -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/cross_encoder/scatter_split/"

echo English/Russian Decoder

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/all/decoder_text/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/cross_decoder/scatter_split/"
