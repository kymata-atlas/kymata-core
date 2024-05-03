import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm

def asr_models_loop_full():

    layer = 34

    neuron = 1280

    size = 'large_v2'
    
    lat_sig = np.zeros((1, layer, neuron, 5)) # ( model, layer, neuron, (peak lat, peak corr, ind, -log(pval), layer_no) )

    log_dir = f'/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/whisper_{size}_multi_log/encoder_all_der_5/'

    n = 1

    for i in range(layer):
        file_name = f'slurm_log_{i}.txt'
        with open(log_dir + file_name, 'r') as f:
            a = f.readlines()
            for ia in range(len(a)):
                if 'encoder' in a[ia]:
                    for k in range(neuron):
                        _a = [j for j in a[ia].split()]
                        lat_sig[i % n, i // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), i // n]
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
    col_2 = lat_sig[:, :, 2]
    col_3 = lat_sig[:, :, 3]
    unique_values = np.unique(col_2)
    max_indices = []
    for val in unique_values:
        indices = np.where(col_2 == val)
        col_3_subset = col_3[indices]
        max_index = indices[1][np.argmax(col_3_subset)]
        max_indices.append(max_index)
    lat_sig = lat_sig[:, max_indices, :]

    # import ipdb;ipdb.set_trace()

    thres = 20

    #lat_i = np.argmax(lat_sig[i, :, :, 3], axis=1)
    #print(lat_i)
    #print(lat_sig[i, 0])
    # _lats = np.array([lat_sig[i, j, lat_i[j], :] for j in range(lat_sig.shape[1]) if lat_sig[i, j, lat_i[j], 0] != 0])

    # import ipdb;ipdb.set_trace()
    _lats = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])
    stds.append(np.std(_lats[:, 0]))

    # import ipdb;ipdb.set_trace()

    scatter = ax.scatter(_lats[:, 0], _lats[:, 4], c= _lats[:, 4], cmap='brg', marker='.', s=15)
    cbar = plt.colorbar(scatter, ax=ax, label='layers')
    # ax.scatter(lat_sig[i, :1, 0], lat_sig[i, :1, 3], marker='o')
    #for j in range(_lats.shape[0]):
    #    ax.annotate(j+1, (_lats[j, 0], _lats[j, 3]))

    plt.ylabel('Layer number')
    plt.xlabel('Latencies (ms)')
    plt.title(f'Threshold -log(p-value): {thres}')
    plt.xlim(-200, 800)
    # plt.legend()
    # plt.xlim(-10, 60)
    plt.savefig(f'kymata-toolbox-data/output/scatter_plot/whisper_encoder_{size}_colour_layer.png', dpi=600)

if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()