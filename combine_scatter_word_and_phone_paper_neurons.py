import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist
from kymata.io.nkg import load_expression_set

def asr_models_loop_full():

    layer = 33 # 66 64 34
    neuron = 4096
    thres = 20 # 15

    phone_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_phone_pvalue.npy'
    word_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_word_with_class_pvalue.npy'

    phone_feat = np.load(phone_dir)
    word_dir = np.load(word_dir)

    fig, ax = plt.subplots()

    # Precompute the max values for each neuron/layer
    max_phone_feat = np.max(phone_feat, axis=0)  # Shape: (neuron, layer)
    max_word_dir = np.max(word_dir, axis=0)      # Shape: (neuron, layer)

    # Create a mask for phoneme-related and word-related neurons
    phoneme_related_mask = max_phone_feat > max_word_dir  # Shape: (neuron, layer)

    # Generate the coordinates for neurons and layers
    x, y = np.meshgrid(np.arange(neuron), np.arange(layer), indexing='ij')  # Shape: (neuron, layer)

    # Scatter plot for phoneme-related neurons
    ax.scatter(x[phoneme_related_mask], y[phoneme_related_mask], c='green', marker='.', s=1, label='Salmonn neurons (phoneme-related)')

    # Scatter plot for word-related neurons
    ax.scatter(x[~phoneme_related_mask], y[~phoneme_related_mask], c='red', marker='.', s=1, label='Salmonn neurons (word-related)')

    plt.xlabel('Neuron number')

    ax.set_ylim(-1, layer)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=5)

    plt.ylabel('Salmonn layer number')

    handles, labels = plt.gca().get_legend_handles_labels()

    # Create a dictionary to remove duplicates (preserve the order)
    unique_labels = dict(zip(labels, handles))

    # Create the legend with unique labels
    ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=5)

    # plt.title(f'Threshold -log(p-value): {thres}')
    # plt.xlim(-200, neuron+200)
    plt.xlim(2000, 2100)
    plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/scatter/fig2a', dpi=600, bbox_inches="tight")


if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()