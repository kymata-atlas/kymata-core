import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist
from kymata.io.nkg import load_expression_set

def generate_green_variations(n):
    green_shades = [(0, i, 0) for i in np.linspace(0, 1, n)]  # varying the green channel
    return green_shades

def generate_red_variations(n):
    red_shades = [(i, 0, 0) for i in np.linspace(0, 1, n)]
    return red_shades

def read_log_file_asr(n, log_dir, layer, neuron):

    lat_sig = np.zeros((n, layer, neuron, 6)) # ( model, layer, neuron, (peak lat, peak corr, ind, -log(pval), layer_no, neuron_no) )

    for i in range(layer):
        file_name = f'slurm_log_{i}.txt'
        with open(log_dir + file_name, 'r') as f:
            a = f.readlines()
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

    return lat_sig

def selection(lat_sig, neuron_selection, layer):
    col_2 = lat_sig[:, 2]
    col_3 = lat_sig[:, 3]
    unique_values = np.unique(col_2)
    max_indices = []
    if 'all' in neuron_selection:
        for val in unique_values:
            indices = np.where(col_2 == val)
            col_3_subset = col_3[indices]
            max_index = indices[0][np.argmax(col_3_subset)]
            max_indices.append(max_index)
    else:
        for val in unique_values:
            for i in range(layer):
                indices = np.where(np.logical_and(col_2 == val, lat_sig[:, 4] == i))
                col_3_subset = col_3[indices]
                try:
                    max_index = indices[0][np.argmax(col_3_subset)]
                    max_indices.append(max_index)
                except:
                    pass
    return lat_sig[max_indices, :]

def asr_models_loop_full():

    layer = 33 # 66 64 34
    neuron = 4096
    neuron_selection = 'layer_sep'
    margin = 0
    n = 1
    figure_opt = 'morpheme'
    thres_feats = 0.01

    log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_7b_morpheme/log/'

    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*370*neuron*layer))))) # maybe we should get rid of the 2 here because we don't split the hemispheres

    stds = []

    lat_sig = read_log_file_asr(n, log_dir, layer, neuron)

    _lats = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])
    # _lats : (point, (latency, corr, sensor, -log(pval), layer, neuron))
    stds.append(np.std(_lats[:, 0]))

    print(_lats.shape[0])


    selected = selection(_lats, neuron_selection, layer)

    print(selected.shape[0])


    neuron_picks = []

    phone_feats = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/phone_sig.npy').tolist()
    word_feats = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/word_sig.npy').tolist()

    feats_path = f'/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_is_root_pvalue.npy'
    feats = np.load(feats_path)

    counter = 0

    for i in range(selected.shape[0]):
        dim = int(selected[i, 5])
        lay = int(selected[i, 4])
        if np.min(feats[:, dim, lay]) < thres_feats and [lay, dim] not in word_feats and [lay, dim] not in phone_feats:
            print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with word feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])}')
            neuron_picks.append([lay, dim])
            counter += 1
    print(f'Number of significant is_root neurons: {counter}')
    np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/morpheme_sig.npy', np.array(neuron_picks))

    feats_path = f'/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_is_prefix_pvalue.npy'
    feats = np.load(feats_path)

    counter = 0

    for i in range(selected.shape[0]):
        dim = int(selected[i, 5])
        lay = int(selected[i, 4])
        if np.min(feats[:, dim, lay]) < thres_feats and [lay, dim] not in neuron_picks and [lay, dim] not in word_feats and [lay, dim] not in phone_feats:
            print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with word feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])}')
            neuron_picks.append([lay, dim])
            counter += 1
    print(f'Number of significant is_prefix neurons: {counter}')
    np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/morpheme_sig_prefix.npy', np.array(neuron_picks))

    feats_path = f'/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_is_suffix_pvalue.npy'
    feats = np.load(feats_path)

    counter = 0

    for i in range(selected.shape[0]):
        dim = int(selected[i, 5])
        lay = int(selected[i, 4])
        if np.min(feats[:, dim, lay]) < thres_feats and [lay, dim] not in neuron_picks and [lay, dim] not in word_feats and [lay, dim] not in phone_feats:
            print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with word feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])}')
            neuron_picks.append([lay, dim])
            counter += 1
    print(f'Number of significant is_suffix neurons: {counter}')
    np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/morpheme_sig_suffix.npy', np.array(neuron_picks))



if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()