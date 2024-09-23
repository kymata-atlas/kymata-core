import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist

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

    neuron_selection = 'all_comb'

    x_data = 'latency'

    margin = 0.1
    
    n = 1

    option = 'plot'
    
    lat_sig = np.zeros((n, layer, neuron, 6)) # ( model, layer, neuron, (peak lat, peak corr, ind, -log(pval), layer_no, neuron_no) )

    log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_7b_phone/log/'

    compare_log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/salmonn_7b/fc2/log/'
    
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

    print(lat_sig.shape)

    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (2*200*370*neuron*layer))))) # maybe we should get rid of the 2 here because we don't split the hemispheres
    thres_tvl = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (2*200*11*neuron*layer)))))

    plt.figure(3)
    fig, ax = plt.subplots()

    stds = []

    lat_sig = lat_sig.reshape(lat_sig.shape[0], -1, lat_sig.shape[3])

    _lats = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])
    # _lats : (point, (latency, corr, sensor, -log(pval), layer, neuron))
    stds.append(np.std(_lats[:, 0]))

    ## Now get the base functions to compare with

    lat_sig = np.zeros((n, layer, neuron, 6))
    for i in range(layer):
        file_name = f'slurm_log_{i}.txt'
        with open(compare_log_dir + file_name, 'r') as f:
            a = f.readlines()
            for ia in range(len(a)):
                if 'layer' in a[ia] and 'Functions to be tested' not in a[ia]:
                    for k in range(neuron):
                        _a = [j for j in a[ia].split()]
                        try:
                            lat_sig[i % n, i // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), i // n, float(_a[0].split('_')[-1].rstrip(':'))]
                        except:
                            pass
                        ia += 1
                    break        
    lat_sig = lat_sig.reshape(lat_sig.shape[0], -1, lat_sig.shape[3])

    _lats_base = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])

    overlap_1 = np.array([_lats[i, :] for i in range(_lats.shape[0]) if _lats[i, -2:].tolist() in _lats_base[:, -2:].tolist()])
    overlap_2 = np.array([_lats_base[i, :] for i in range(_lats_base.shape[0]) if _lats_base[i, -2:].tolist() in _lats[:, -2:].tolist()])
    enhanced = np.array([overlap_1[i, :] for i in range(overlap_1.shape[0]) if overlap_1[i, 3] >= overlap_2[i, 3] * (1 + margin)])
    reduced  = np.array([overlap_1[i, :] for i in range(overlap_1.shape[0]) if overlap_2[i, 3] >= overlap_1[i, 3] * (1 + margin)])
    emerge = np.array([_lats[i, :] for i in range(_lats.shape[0]) if _lats[i, :].tolist() not in overlap_1.tolist()])
    demolish = np.array([_lats_base[i, :] for i in range(_lats_base.shape[0]) if _lats_base[i, :].tolist() not in overlap_2.tolist()])

    if neuron_selection != 'no':
        if 'sep' in neuron_selection:
            enhanced = selection(enhanced, neuron_selection, layer)
            reduced = selection(reduced, neuron_selection, layer)
            emerge = selection(emerge, neuron_selection, layer)
            demolish = selection(demolish, neuron_selection, layer)
        else:
            # Adding a label for phone (0) and word (1)
            enhanced = np.hstack((enhanced, np.zeros((enhanced.shape[0], 1))))
            emerge = np.hstack((emerge, np.zeros((emerge.shape[0], 1))))
            reduced = np.hstack((reduced, np.ones((reduced.shape[0], 1))))
            demolish = np.hstack((demolish, np.ones((demolish.shape[0], 1))))
            all_data = np.vstack((enhanced, emerge, reduced, demolish))
            all_data = selection(all_data, neuron_selection, layer)
            
    # import ipdb;ipdb.set_trace()

    if option == 'plot':

        if 'sep' in neuron_selection: 

            if x_data == 'latency':
                scatter = ax.scatter(enhanced[:, 0], enhanced[:, 4], c='green', marker='.', s=15, label = 'Phone')
                scatter = ax.scatter(reduced[:, 0], reduced[:, 4], c='red', marker='.', s=15, label = 'Word')
                scatter = ax.scatter(emerge[:, 0], emerge[:, 4], c='green', marker='.', s=15)
                scatter = ax.scatter(demolish[:, 0], demolish[:, 4], c='red', marker='.', s=15)
                plt.xlabel('Latencies (ms)')
            else:
                x_upper = neuron + 200
                scatter = ax.scatter(enhanced[:, 5], enhanced[:, 4], c='green', marker='.', s=15, label = 'Phone')
                scatter = ax.scatter(reduced[:, 5], reduced[:, 4], c='red', marker='.', s=15, label = 'Word')
                scatter = ax.scatter(emerge[:, 5], emerge[:, 4], c='green', marker='.', s=15)
                scatter = ax.scatter(demolish[:, 5], demolish[:, 4], c='red', marker='.', s=15)
                plt.xlabel('Neuron number')

        else:

            mask_1 = np.array([i for i in range(all_data.shape[0]) if all_data[i, -1] == 0])
            mask_2 = np.array([i for i in range(all_data.shape[0]) if all_data[i, -1] == 1])
            if x_data == 'latency':
                scatter = ax.scatter(all_data[mask_1, 0], all_data[mask_1, 4], c='green', marker='.', s=15, label = 'Phone')
                scatter = ax.scatter(all_data[mask_2, 0], all_data[mask_2, 4], c='red', marker='.', s=15, label = 'Word')
                plt.xlabel('Latencies (ms)')
            else:
                scatter = ax.scatter(all_data[mask_1, 5], all_data[mask_1, 4], c='green', marker='.', s=15, label = 'Phone')
                scatter = ax.scatter(all_data[mask_2, 5], all_data[mask_2, 4], c='red', marker='.', s=15, label = 'Word')
                plt.xlabel('Neuron number')

        ax.set_ylim(-1, layer)
        ax.legend()


        plt.ylabel('Layer number')
        plt.title(f'Threshold -log(p-value): {thres}')
        plt.xlim(-200, x_upper)
        plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/scatter/salmonn_7b_phone_vs_word_{neuron_selection}_select', dpi=600)

    else:

        pass

if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()