import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist
from kymata.io.nkg import load_expression_set

def read_log_file_asr(n, log_dir, layer, neuron):

    lat_sig = np.zeros((n, layer, neuron, 6)) # ( model, layer, neuron, (peak lat, peak corr, ind, -log(pval), layer_no, neuron_no) )

    for i in range(layer):
        file_name = f'slurm_log_{i}.txt'
        with open(log_dir + file_name, 'r') as f:
            a = f.readlines()
            for ia in range(len(a)):
                if 'layer' in a[ia] and 'Functions to be tested' not in a[ia]:
                    for k in range(neuron):
                        _a = [j for j in a[ia].split()]
                        lat_sig[i % n, i // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), i // n, float(_a[0].split('_')[-1].rstrip(':'))]
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
    thres = 20 # 15
    x_upper = 800
    x_data = 'latency'
    neuron_selection = 'layer_sep'
    margin = 0
    n = 1
    figure_opt = 'phone'

    log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_7b_phone/fc2/log/'
    compare_log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/salmonn_7b/fc2/log/'
    tvl_compare_log_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/salmonn_7b/tvl/log/'
    tvl_log_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_7b_phone/tvl/log/'

    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*370*neuron*layer))))) # maybe we should get rid of the 2 here because we don't split the hemispheres
    thres_tvl = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*11*neuron*layer)))))
    thres_tvl_true = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*11*370)))))
    thres_feats = 0.05

    plt.figure(3)
    fig, ax = plt.subplots()

    stds = []

    lat_sig = read_log_file_asr(n, log_dir, layer, neuron)

    _lats = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])
    # _lats : (point, (latency, corr, sensor, -log(pval), layer, neuron))
    stds.append(np.std(_lats[:, 0]))

    ## Now get the base functions to compare with

    lat_sig = read_log_file_asr(n, compare_log_dir, layer, neuron)

    _lats_base = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])

    overlap_1 = np.array([_lats[i, :] for i in range(_lats.shape[0]) if _lats[i, -2:].tolist() in _lats_base[:, -2:].tolist()])
    overlap_2 = np.array([_lats_base[i, :] for i in range(_lats_base.shape[0]) if _lats_base[i, -2:].tolist() in _lats[:, -2:].tolist()])
    enhanced = np.array([overlap_1[i, :] for i in range(overlap_1.shape[0]) if overlap_1[i, 3] >= overlap_2[i, 3] * (1 + margin)])
    reduced  = np.array([overlap_1[i, :] for i in range(overlap_1.shape[0]) if overlap_2[i, 3] >= overlap_1[i, 3] * (1 + margin)])
    emerge = np.array([_lats[i, :] for i in range(_lats.shape[0]) if _lats[i, :].tolist() not in overlap_1.tolist()])
    demolish = np.array([_lats_base[i, :] for i in range(_lats_base.shape[0]) if _lats_base[i, :].tolist() not in overlap_2.tolist()])

    enhanced = selection(enhanced, neuron_selection, layer)
    reduced = selection(reduced, neuron_selection, layer)
    emerge = selection(emerge, neuron_selection, layer)
    demolish = selection(demolish, neuron_selection, layer)

    if x_data == 'latency':

        lat_sig = read_log_file_asr(n, tvl_log_dir, layer, neuron)
        _lats_tvl = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)])
        lat_sig = read_log_file_asr(n, tvl_compare_log_dir, layer, neuron)
        _lats_tvl_base = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)])

        mask_phone_enhanced = np.array([i for i in range(enhanced.shape[0]) if not np.any(np.all(enhanced[i, 4:] == _lats_tvl[:, 4:], axis=1))])
        mask_phone_emerge = np.array([i for i in range(emerge.shape[0]) if not np.any(np.all(emerge[i, 4:] == _lats_tvl[:, 4:], axis=1))])
        mask_phone_reduced = np.array([i for i in range(reduced.shape[0]) if not np.any(np.all(reduced[i, 4:] == _lats_tvl_base[:, 4:], axis=1))])
        mask_phone_demolish = np.array([i for i in range(demolish.shape[0]) if not np.any(np.all(demolish[i, 4:] == _lats_tvl_base[:, 4:], axis=1))])

        # scatter = ax.scatter(reduced[mask_phone_reduced, 0], reduced[mask_phone_reduced, 4], c='red', marker='.', s=15, label = 'Salmonn neurons (word-related)')
        scatter = ax.scatter(enhanced[mask_phone_enhanced, 0], enhanced[mask_phone_enhanced, 4], c='green', marker='.', s=15, label = 'Salmonn neurons (phone-related)')
        scatter = ax.scatter(emerge[mask_phone_emerge, 0], emerge[mask_phone_emerge, 4], c='green', marker='.', s=15)
        # scatter = ax.scatter(demolish[mask_phone_demolish, 0], demolish[mask_phone_demolish, 4], c='red', marker='.', s=15)

        if figure_opt == 'phone':
            feats_path = '/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_phone_pvalue.npy'
            feats = np.load(feats_path)
            # import ipdb;ipdb.set_trace()
            counter = 0
            for i in range(mask_phone_enhanced.shape[0]):
                dim = int(enhanced[mask_phone_enhanced[i], 5])
                layer = int(enhanced[mask_phone_enhanced[i], 4])
                if np.min(feats[:, dim, layer]) < thres_feats:
                    print(f'The Salmonn neuron {dim} at layer {layer} has the most significant correlation with phonetic feature {np.argmin(feats[:, dim, layer])} with a p-value of {np.min(feats[:, dim, layer])}')
                    counter += 1
            for i in range(mask_phone_emerge.shape[0]):
                dim = int(emerge[mask_phone_emerge[i], 5])
                layer = int(emerge[mask_phone_emerge[i], 4])
                if np.min(feats[:, dim, layer]) < thres_feats:
                    print(f'The Salmonn neuron {dim} at layer {layer} has the most significant correlation with phonetic feature {np.argmin(feats[:, dim, layer])} with a p-value of {np.min(feats[:, dim, layer])}')
                    counter += 1
            print(f'Proportion of significant neurons: {counter/(mask_phone_enhanced.shape[0]+mask_phone_emerge.shape[0])}')
        else:
            feats_path = '/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_word_pvalue.npy'
            feats = np.load(feats_path)
            feats[7:11, :, :] = np.ones((4, dim, layer))
            for i in range(mask_phone_reduced.shape[0]):
                dim = int(reduced[mask_phone_reduced[i], 5])
                layer = int(reduced[mask_phone_reduced[i], 4])
                if np.min(feats[:, dim, layer]) < thres_feats:
                    print(f'The Salmonn neuron {dim} at layer {layer} has the most significant correlation with word feature {np.argmin(feats[:, dim, layer])} with a p-value of {np.min(feats[:, dim, layer])}')
            for i in range(mask_phone_demolish.shape[0]):
                dim = int(demolish[mask_phone_demolish[i], 5])
                layer = int(demolish[mask_phone_demolish[i], 4])
                if np.min(feats[:, dim, layer]) < thres_feats:
                    print(f'The Salmonn neuron {dim} at layer {layer} has the most significant correlation with word feature {np.argmin(feats[:, dim, layer])} with a p-value of {np.min(feats[:, dim, layer])}')
         

        plt.xlabel('Latency (ms) relative to onset of the environment')

    else:

        x_upper = neuron + 200
        scatter = ax.scatter(enhanced[:, 5], enhanced[:, 4], c='green', marker='.', s=15, label = 'Salmonn neurons (phone-related)')
        scatter = ax.scatter(reduced[:, 5], reduced[:, 4], c='red', marker='.', s=15, label = 'Salmonn neurons (word-related)')
        scatter = ax.scatter(emerge[:, 5], emerge[:, 4], c='green', marker='.', s=15)
        scatter = ax.scatter(demolish[:, 5], demolish[:, 4], c='red', marker='.', s=15)
        plt.xlabel('Neuron number')

    ax.set_ylim(-1, layer)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=5)
    ax.axvline(x=0, color='k', linestyle='dotted')


    plt.ylabel('Salmonn layer number')

    # # Define y-tick positions and labels
    # yticks = [0, 5, 10, 15, 20, 25, 30]
    # ytick_labels = ['TVL-related functions', 5, 10, 15, 20, 25, 30]

    # # Set y-ticks and custom labels
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(ytick_labels)

    # plt.title(f'Threshold -log(p-value): {thres}')
    plt.xlim(-200, x_upper)
    plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/scatter/salmonn_7b_phone_vs_word_v4', dpi=600, bbox_inches="tight")


if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()