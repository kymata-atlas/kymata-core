import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm

def asr_models_loop_full():

    reindex = False

    language = 'en'

    layer = 66

    neuron = 1280

    thres = 15

    size = 'large'

    neuron_selection = True
    
    n = 1
    
    lat_sig = np.zeros((n, layer, neuron, 6)) # ( model, layer, neuron, (peak lat, peak corr, ind, -log(pval), layer_no, neuron_no) )

    if language == 'en':
        log_dir_1 = f'/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/whisper_{size}_multi_log/encoder_all_der_5/'
        log_dir_2 = f'/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/whisper_{size}_multi_log/decoder_all_der_5/'
    else:
        log_dir_1 = f'/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/whisper_large_encoder_log/'
        log_dir_2 = f'/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/russian/whisper_large_decoder_log/'

    for i in range(layer-32):
        file_name = f'slurm_log_{i}.txt'
        with open(log_dir_1 + file_name, 'r') as f:
            a = f.readlines()
            for ia in range(len(a)):
                if 'model' in a[ia]:
                    for k in range(neuron):
                        _a = [j for j in a[ia].split()]
                        lat_sig[i % n, i // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), i // n, float(_a[0].split('_')[-1].rstrip(':'))]
                        ia += 1
                    break

    for i in range(layer-34):
        file_name = f'slurm_log_{i}.txt'
        with open(log_dir_2 + file_name, 'r') as f:
            a = f.readlines()
            for ia in range(len(a)):
                if 'model' in a[ia]:
                    for k in range(neuron):
                        _a = [j for j in a[ia].split()]
                        lat_sig[(i+34) % n, (i+34) // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), (i+34) // n, float(_a[0].split('_')[-1].rstrip(':'))]
                        ia += 1
                    break

    print(lat_sig[0, 0])
    print(lat_sig.shape)

    plt.figure(3)
    fig, ax = plt.subplots()

    stds = []

    lat_sig = lat_sig.reshape(lat_sig.shape[0], -1, lat_sig.shape[3])

    # import ipdb;ipdb.set_trace()

    # Neuron selection
    if neuron_selection:
        col_2 = lat_sig[:, (lat_sig[0, :, 4]<34), 2]
        col_3 = lat_sig[:, (lat_sig[0, :, 4]<34), 3]
        unique_values = np.unique(col_2)
        max_indices = []
        for val in unique_values:
            indices = np.where(col_2 == val)
            col_3_subset = col_3[indices]
            max_index = indices[1][np.argmax(col_3_subset)]
            max_indices.append(max_index)
        lat_sig_max = lat_sig[:, max_indices, :]
    lat_sig_dec = lat_sig[:, (lat_sig[0, :, 4]>33), :]
    lat_sig = np.concatenate((lat_sig_max, lat_sig_dec), axis = 1)
    # import ipdb;ipdb.set_trace()

    #lat_i = np.argmax(lat_sig[i, :, :, 3], axis=1)
    #print(lat_i)
    #print(lat_sig[i, 0])
    # _lats = np.array([lat_sig[i, j, lat_i[j], :] for j in range(lat_sig.shape[1]) if lat_sig[i, j, lat_i[j], 0] != 0])

    # import ipdb;ipdb.set_trace()
    _lats = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])
    stds.append(np.std(_lats[:, 0]))

    # import ipdb;ipdb.set_trace()

    # Extract the unique group values from _lats[:, 4]
    unique_groups = np.unique(_lats[:, 4])

    # Initialize lists to hold the x and y values for the scatter plot
    x_values = []
    y_values = []
    colors = []
    grey_dot_alpha = 0.3


    # Define the colors based on the conditions
    color_map = {0: 'red', 1: 'red',  33: 'green', 34: 'green', 65: 'purple'}
    colored_dot_size = 30  # Size for colored dots
    grey_dot_size = 15  # Size for grey dots

    # Iterate over each unique group
    for group in unique_groups:
        # Get the indices of the entries that belong to the current group
        group_indices = np.where(_lats[:, 4] == group)[0]
        
        # Iterate over the indices and assign y values based on the index within the group
        for idx, original_idx in enumerate(group_indices):
            x_values.append(_lats[original_idx, 0])  # Assuming you want to plot _lats[:, 0] on the x-axis
            if reindex:
                y_values.append(idx)  # The y value is the index within the group
            else:
                y_values.append(_lats[original_idx, -1])
            # import ipdb;ipdb.set_trace()
            colors.append(color_map.get(int(group), 'grey'))

    # import ipdb;ipdb.set_trace()

    # scatter = ax.scatter(_lats[:, 0], y_values, c= _lats[:, 4], cmap='brg', marker='.', s=15)
    scatter = plt.scatter(x_values, y_values, c=colors, s=[colored_dot_size if c != 'grey' else grey_dot_size for c in colors], marker='.', alpha=[1 if c != 'grey' else grey_dot_alpha for c in colors])
    # cbar = plt.colorbar(scatter, ax=ax, label='layers')
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Layer {key}', markerfacecolor=value, markersize=10) for key, value in color_map.items()]
    plt.legend(handles=legend_elements)
    # ax.scatter(lat_sig[i, :1, 0], lat_sig[i, :1, 3], marker='o')
    #for j in range(_lats.shape[0]):
    #    ax.annotate(j+1, (_lats[j, 0], _lats[j, 3]))

    plt.ylabel('Neuron number within a layer')
    plt.xlabel('Latencies (ms)')
    plt.title(f'Threshold -log(p-value): {thres}')
    plt.xlim(-200, 800)
    # plt.legend()
    # plt.xlim(-10, 60)
    if language == 'en':
        plt.savefig(f'kymata-toolbox-data/output/scatter_plot/whisper_all_{size}_colour_layer.png', dpi=600)
    else:
        plt.savefig(f'kymata-toolbox-data/output/scatter_plot/ru_whisper_all_{size}_colour_layer.png', dpi=600)

if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()