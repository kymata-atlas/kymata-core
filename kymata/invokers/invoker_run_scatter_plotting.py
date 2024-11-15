from logging import basicConfig, INFO
from pathlib import Path
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist

from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set
from kymata.plot.plot import expression_plot, legend_display_dict
from kymata.plot.color import constant_color_dict, gradient_color_dict

import time
from tqdm import tqdm

def load_all_expression_data(base_folder):
    expression_data = None
    # Loop through each subdirectory inside the base folder
    for subdir in os.listdir(base_folder):
        subdir_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subdir_path):  # Ensure we are processing directories
            # List all .nkg files inside the subdirectory
            nkg_files = [f for f in os.listdir(subdir_path) if f.endswith('.nkg')]
            for nkg_file in nkg_files:
                file_path = os.path.join(subdir_path, nkg_file)
                if expression_data is None:
                    # Load the first .nkg file
                    expression_data = load_expression_set(file_path)
                else:
                    # Add data from subsequent .nkg files
                    expression_data += load_expression_set(file_path)
    return expression_data

def load_part_of_expression_data(base_folder, pick):
    expression_data = None
    # Loop through each subdirectory inside the base folder
    for subdir in os.listdir(base_folder):
        subdir_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subdir_path) and [int(x) for x in subdir.split('_')] in pick.tolist():  # Ensure we are processing directories
            # List all .nkg files inside the subdirectory
            nkg_files = [f for f in os.listdir(subdir_path) if f.endswith('.nkg')]
            for nkg_file in nkg_files:
                file_path = os.path.join(subdir_path, nkg_file)
                if expression_data is None:
                    # Load the first .nkg file
                    expression_data = load_expression_set(file_path)
                else:
                    # Add data from subsequent .nkg files
                    expression_data += load_expression_set(file_path)
    return expression_data

def main():

    # expression_data_salmonn_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron')
    expression_data_salmonn_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/word_source')
    word_name = expression_data_salmonn_word.transforms
    # expression_data_salmonn_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron_phone')
    # expression_data_salmonn_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/phone_source')
    # phone_name = expression_data_salmonn_phone.transforms
    phone_name = []
    expression_data_tvl = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/english_TVL_family_source_baseline_derangments_6.nkg')
    tvl_name = expression_data_tvl.transforms
    IL_name = [i for i in tvl_name if i != 'STL']
    STL_name = ['STL']
    # fig = expression_plot(expression_data_salmonn_word + expression_data_tvl + expression_data_salmonn_phone, paired_axes=True, minimap=False, show_legend=True,
    #                         color=constant_color_dict(word_name, color= 'red')
    #                             # | constant_color_dict(tvl_name, color= 'yellow')
    #                             | constant_color_dict(IL_name, color= 'purple')
    #                             | constant_color_dict(STL_name, color= 'pink')
    #                             | constant_color_dict(phone_name, color='green'),
    #                         legend_display=legend_display_dict(word_name, 'SALMONN word features')
    #                             # | legend_display_dict(tvl_name, 'TVL transforms')
    #                             | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
    #                             | legend_display_dict(STL_name, 'Short Term Loudness transform')
    #                             | legend_display_dict(phone_name, 'SALMONN phone features'))

    # all_expression_data = expression_data_salmonn_word + expression_data_tvl + expression_data_salmonn_phone
    all_expression_data = expression_data_salmonn_word + expression_data_tvl
    data = all_expression_data.best_transforms()
    n_channels = len(all_expression_data.hexels_left) + len(all_expression_data.hexels_right)
    n_transforms = len(all_expression_data.transforms)
    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*n_channels*n_transforms)))))

    plt.figure(figsize=(10, 6))

    # import ipdb;ipdb.set_trace()

    for i in tqdm(range(data[0].values.shape[0])):
        if -data[0].values[i][-1] > thres:
            if 'layer' in data[0].values[i][1]:
                if data[0].values[i][1] in word_name:
                    plt.scatter(data[0].values[i][2]*1000, int(data[0].values[i][1].split('_')[0][5:]) + 1, c='red', marker='.', s=15, alpha=0.7, label = 'Salmonn word features')
                elif data[0].values[i][1] in phone_name:
                    plt.scatter(data[0].values[i][2]*1000, int(data[0].values[i][1].split('_')[0][5:]) + 1, c='green', marker='.', s=15, alpha=0.7, label = 'Salmonn phoneme features')
            elif data[0].values[i][1] in STL_name:
                plt.scatter(data[0].values[i][2]*1000, 0, c='pink', marker='.', s=15, alpha=0.7, label = 'Perceptual-Loudness-related (Moore et al., 2001)')
            elif data[0].values[i][1] in IL_name:
                plt.scatter(data[0].values[i][2]*1000, 0, c='purple', marker='.', s=15, alpha=0.7, label = 'Instantaneous-Loudness-related (Moore et al., 2001)')

    for i in tqdm(range(data[1].values.shape[0])):
        if -data[0].values[i][-1] > thres:
            if 'layer' in data[0].values[i][1]:
                if data[0].values[i][1] in word_name:
                    plt.scatter(data[0].values[i][2]*1000, int(data[0].values[i][1].split('_')[0][5:]) + 1, c='red', marker='.', s=15, alpha=0.7, label = 'Salmonn word features')
                elif data[0].values[i][1] in phone_name:
                    plt.scatter(data[0].values[i][2]*1000, int(data[0].values[i][1].split('_')[0][5:]) + 1, c='green', marker='.', s=15, alpha=0.7, label = 'Salmonn phoneme features')
            elif data[0].values[i][1] in STL_name:
                plt.scatter(data[0].values[i][2]*1000, 0, c='pink', marker='.', s=15, alpha=0.7, label = 'Perceptual-Loudness-related (Moore et al., 2001)')
            elif data[0].values[i][1] in IL_name:
                plt.scatter(data[0].values[i][2]*1000, 0, c='purple', marker='.', s=15, alpha=0.7, label = 'Instantaneous-Loudness-related (Moore et al., 2001)')

    plt.xlabel('Latency (ms) relative to onset of the environment')
    plt.ylabel('Salmonn layer number')

    # Define y-tick positions and labels
    yticks = [0, 5, 10, 15, 20, 25, 30]
    ytick_labels = ['TVL-related functions', 5, 10, 15, 20, 25, 30]

    # Set y-ticks and custom labels
    plt.yticks(yticks, ytick_labels)

    # Get the current handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Create a dictionary to filter out duplicate labels
    unique_labels = dict(zip(labels, handles))

    # Set the legend with unique labels
    plt.legend(unique_labels.values(), unique_labels.keys(), bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=5)

    # Add padding around the data points
    plt.margins(x=0.05, y=0.05)

    # Automatically adjust subplot parameters to give some padding
    plt.tight_layout()

    plt.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_word_vs_tvl_all_scatter.png")

if __name__ == '__main__':
    main()