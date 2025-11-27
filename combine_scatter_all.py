import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist

def process_log(layer, neuron, n, log_dir, neuron_selection, thres):

    lat_sig = np.zeros((n, layer, neuron, 6))  # ( model, layer, neuron, (peak lat, peak corr, ind, -log(pval), layer_no, neuron_no) )

    for i in range(layer):
        file_name = f'slurm_log_{i}.txt'
        # file_name = f'slurm_log_{i}.txt'
        with open(log_dir + file_name, 'r') as f:
            a = f.readlines()
            a = [line for line in a if 'Time' not in line]
            for ia in range(len(a)):
                if 'layer' in a[ia] and 'Functions to be tested' not in a[ia] and 'Transforms to be tested' not in a[ia]:
                    for k in range(neuron):
                        _a = [j for j in a[ia].split()]
                        try:
                            lat_sig[i % n, i // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), i // n, float(_a[0].split('_')[-1].rstrip(':'))]
                        except:
                            lat_sig[i % n, i // n, k] = [float(_a[10][:-1]), 0, float(_a[16][:-1]), float(_a[-1]), i // n, float(_a[7].split('_')[-1].rstrip(':'))]
                        ia += 1
                    break

    lat_sig = lat_sig.reshape(lat_sig.shape[0], -1, lat_sig.shape[3])

    # Neuron selection
    if neuron_selection != 'no':
        col_2 = lat_sig[:, :, 2]
        col_3 = lat_sig[:, :, 3]
        unique_values = np.unique(col_2)
        max_indices = []
        if neuron_selection == 'all':
            for val in unique_values:
                indices = np.where(col_2 == val)
                col_3_subset = col_3[indices]
                max_index = indices[1][np.argmax(col_3_subset)]
                max_indices.append(max_index)
        else:
            for val in unique_values:
                for i in range(layer):
                    # import ipdb;ipdb.set_trace()
                    indices = np.where(np.logical_and(col_2 == val, lat_sig[:, :, 4] == i))
                    col_3_subset = col_3[indices]
                    try:
                        max_index = indices[1][np.argmax(col_3_subset)]
                        max_indices.append(max_index)
                    except:
                        pass
        lat_sig = lat_sig[:, max_indices, :]

    _lats = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])
    
    return _lats
    

def asr_models_loop_full():

    layer = 33 # 41 # 66 64 34 33

    neuron = 4096 # 4096 5120

    thres = 20 # 15

    x_upper = 800

    neuron_selection = 'layer'
 
    n = 1
    
    log_dir_morpheme = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_morpheme/log/'
    log_dir_word = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_word/log/'
    log_dir_phone = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_phone/log/'
    log_dir_wordpiece = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_wordpiece/log/'

    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (2*200*370*neuron*layer))))) # maybe we should get rid of the 2 here because we don't split the hemispheres

    
    _lats_morpheme = process_log(layer, neuron, n, log_dir_morpheme, neuron_selection, thres)
    _lats_word = process_log(layer, neuron, n, log_dir_word, neuron_selection, thres)
    _lats_phone = process_log(layer, neuron, n, log_dir_phone, neuron_selection, thres)
    _lats_wordpiece = process_log(layer, neuron, n, log_dir_wordpiece, neuron_selection, thres)



    plt.figure(3)
    fig, ax = plt.subplots()


    # scatter = ax.scatter(_lats_morpheme[:, 0], _lats_morpheme[:, 4], c= 'blue', marker='.', s=15)
    scatter = ax.scatter(_lats_word[:, 0], _lats_word[:, 4], c= 'red', marker='.', s=15)
    # scatter = ax.scatter(_lats_phone[:, 0], _lats_phone[:, 4], c= 'green', marker='.', s=15)
    # scatter = ax.scatter(_lats_wordpiece[:, 0], _lats_wordpiece[:, 4], c= 'orange', marker='.', s=15)

    cbar = plt.colorbar(scatter, ax=ax, label='layers')
    ax.set_ylim(-1, layer)

    plt.ylabel('Layer number')
    plt.xlabel('Latencies (ms)')
    plt.title(f'Threshold -log(p-value): {thres}')
    plt.xlim(-200, x_upper)
    plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/new_exp/scatter_{neuron_selection}_word', dpi=600)

if __name__ == '__main__':
    asr_models_loop_full()