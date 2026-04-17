#!/bin/bash

echo English

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/ecog_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/"
poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/eeg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/"
poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/"
poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/ecog_sig_neurons_layer29_neuron3584.npy" \
     "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/eeg_sig_neurons_layer29_neuron3584.npy" \
     "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen/"

echo Russian

poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen_russian/eeg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen_russian/"
poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen_russian/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen_russian/"
poetry run python kymata/invokers/language_neuron_fit.py -m 4 \
  -i "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen_russian/eeg_sig_neurons_layer29_neuron3584.npy" \
     "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen_russian/meg_sig_neurons_layer29_neuron3584.npy" \
  -o "/imaging/projects/cbu/kymata/analyses/cai/kymata-core/kymata-core-data/tianyi/sensor/scatter_split_qwen_russian/"
