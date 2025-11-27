import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist

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
    return lat_sig[:, max_indices, :]

def asr_models_loop_full():

    layer = 33 # 66 64 34

    neuron = 4096

    thres = 20 # 15

    x_upper = 800

    size = '7b_syntax'

    neuron_selection = 'no'

    x_data = 'latency'

    margin = 0.1
    
    n = 1

    option = 'stats'
    
    log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/prompt/{size}/fc2/log/'

    compare_log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/salmonn_7b/fc2/log/'

    tvl_log_base = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/salmonn_7b/tvl/log/'

    tvl_log = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/prompt/{size}/tvl/log/'

    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (2*200*370*neuron*layer)))))
    thres_tvl = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (2*200*11*neuron*layer)))))

    plt.figure(3)
    fig, ax = plt.subplots()

    stds = []

    lat_sig = read_log_file_asr(n, log_dir, layer, neuron)

    if neuron_selection != 'no':
        lat_sig = selection(lat_sig, neuron_selection, layer)

    _lats = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])
    # _lats : (point, (latency, corr, sensor, -log(pval), layer, neuron))
    stds.append(np.std(_lats[:, 0]))


    ## Now get the base functions to compare with

    lat_sig = read_log_file_asr(n, compare_log_dir, layer, neuron)

    if neuron_selection != 'no':
        lat_sig = selection(lat_sig, neuron_selection, layer)

    _lats_base = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])


    overlap_1 = np.array([_lats[i, :] for i in range(_lats.shape[0]) if _lats[i, -2:].tolist() in _lats_base[:, -2:].tolist()])
    overlap_2 = np.array([_lats_base[i, :] for i in range(_lats_base.shape[0]) if _lats_base[i, -2:].tolist() in _lats[:, -2:].tolist()])
    enhanced = np.array([overlap_1[i, :] for i in range(overlap_1.shape[0]) if overlap_1[i, 3] >= overlap_2[i, 3] * (1 + margin)])
    reduced  = np.array([overlap_1[i, :] for i in range(overlap_1.shape[0]) if overlap_2[i, 3] >= overlap_1[i, 3] * (1 + margin)])
    emerge = np.array([_lats[i, :] for i in range(_lats.shape[0]) if _lats[i, :].tolist() not in overlap_1.tolist()])
    demolish = np.array([_lats_base[i, :] for i in range(_lats_base.shape[0]) if _lats_base[i, :].tolist() not in overlap_2.tolist()])

    # import ipdb;ipdb.set_trace()

    if option == 'plot':

        if x_data == 'latency':
            scatter = ax.scatter(enhanced[:, 0], enhanced[:, 4], c='green', marker='.', s=15, label = 'Enhanced')
            scatter = ax.scatter(reduced[:, 0], reduced[:, 4], c='red', marker='.', s=15, label = 'Reduced')
            scatter = ax.scatter(emerge[:, 0], emerge[:, 4], c='blue', marker='.', s=15, label = 'Emerge')
            scatter = ax.scatter(demolish[:, 0], demolish[:, 4], c='black', marker='.', s=15, label = 'Demolish', alpha=0.5)
            plt.xlabel('Latencies (ms)')
        else:
            x_upper = neuron + 200
            scatter = ax.scatter(enhanced[:, 5], enhanced[:, 4], c='green', marker='.', s=15, label = 'Enhanced')
            scatter = ax.scatter(reduced[:, 5], reduced[:, 4], c='red', marker='.', s=15, label = 'Reduced')
            scatter = ax.scatter(emerge[:, 5], emerge[:, 4], c='blue', marker='.', s=15, label = 'Emerge')
            scatter = ax.scatter(demolish[:, 5], demolish[:, 4], c='black', marker='.', s=15, label = 'Demolish', alpha=0.5)
            plt.xlabel('Neuron number')

        ax.set_ylim(-1, layer)
        ax.legend()


        plt.ylabel('Layer number')
        plt.title(f'Threshold -log(p-value): {thres}')
        plt.xlim(-200, x_upper)
        plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/scatter/prompt_{size}_{x_data}_{neuron_selection}_select_new_thres', dpi=600)

    else:

        lat_sig = read_log_file_asr(n, tvl_log_base, layer, neuron)

        _lats_tvl_base = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)])

        reduced_tvl = np.array([reduced[i, :] for i in range(reduced.shape[0]) if reduced[i, -2:].tolist() in _lats_tvl_base[:, -2:].tolist()])
        demolish_tvl = np.array([demolish[i, :] for i in range(demolish.shape[0]) if demolish[i, -2:].tolist() in _lats_tvl_base[:, -2:].tolist()])

        lat_sig = read_log_file_asr(n, tvl_log, layer, neuron)

        _lats_tvl = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)])

        enhanced_tvl = np.array([enhanced[i, :] for i in range(enhanced.shape[0]) if enhanced[i, -2:].tolist() in _lats_tvl[:, -2:].tolist()])
        emerge_tvl = np.array([emerge[i, :] for i in range(emerge.shape[0]) if emerge[i, -2:].tolist() in _lats_tvl[:, -2:].tolist()])

        print(reduced_tvl.shape[0]/reduced.shape[0])
        print(demolish_tvl.shape[0]/demolish.shape[0])
        print(enhanced_tvl.shape[0]/enhanced.shape[0])
        print(emerge_tvl.shape[0]/emerge.shape[0])

        import ipdb;ipdb.set_trace()

if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()