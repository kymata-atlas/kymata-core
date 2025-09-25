import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist

def asr_models_loop_full():

    layer = 33 # 41 # 66 64 34 33

    neuron = 4096 # 4096 5120

    thres = 20 # 15

    x_upper = 800

    size = 'large'

    neuron_selection = 'layer'

    exclude_tvl = True
    
    n = 1
    
    lat_sig = np.zeros((n, layer, neuron, 6)) # ( model, layer, neuron, (peak lat, peak corr, ind, -log(pval), layer_no, neuron_no) )

    # log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/{size}/fc2/log/'
    # log_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/salmonn_omni/sensor/log/'
    log_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_word/log/'

    # log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/{size}/log/'

    # log_tvl_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/{size}/tvl/log/'
    # log_tvl_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/salmonn_omni/sensor/tvl/log/'
    log_tvl_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_word/tvl/log/'


    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (2*200*370*neuron*layer))))) # maybe we should get rid of the 2 here because we don't split the hemispheres
    thres_tvl = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (2*200*11*neuron*layer)))))

    # for i in range(layer):
    #     file_name = f'slurm_log_{i}.txt'
    #     with open(log_dir + file_name, 'r') as f:
    #         a = f.readlines()
    #         for ia in range(len(a)):
    #             if 'model' in a[ia] and 'Functions to be tested' not in a[ia]:
    #                 for k in range(neuron):
    #                     _a = [j for j in a[ia].split()]
    #                     lat_sig[i % n, i // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), i // n, float(_a[0].split('_')[-1].rstrip(':'))]
    #                     ia += 1
    #                 break

    log_dir_1 = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/large/fc2/log/'

    log_dir_2 = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/large/fc2/log/'

    # for i in range(layer-32):
    #     if 'fake' in log_dir_1:
    #         # file_name = f'model.encoder.layers.{i}.fc2_1279_gridsearch_results.txt'
    #         file_name = f'model.encoder.layers.{i}.final_layer_norm_1279_gridsearch_results.txt'
    #     else:
    #         file_name = f'slurm_log_{i}.txt'
    #     with open(log_dir_1 + file_name, 'r') as f:
    #         a = f.readlines()
    #         for ia in range(len(a)):
    #             if 'model' in a[ia] and 'Functions to be tested' not in a[ia]:
    #                 for k in range(neuron):
    #                     _a = [j for j in a[ia].split()]
    #                     try:
    #                         if len(_a) == 12:
    #                             lat_sig[i % n, i // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), i // n, float(_a[0].split('_')[-1].rstrip(':'))]
    #                         else:
    #                             lat_sig[i % n, i // n, k] = [float(_a[3][:-1]), 0, float(_a[6][:-1]), float(_a[-1]), i // n, float(_a[0].split('_')[-1].rstrip(':'))]
    #                     except:
    #                         pass
    #                     ia += 1
    #                 break

    # for i in range(layer-32 if layer == 64 else layer-34): # (layer-34)
    #     if 'fake' in log_dir_1:
    #         # file_name = f'model.decoder.layers.{i}.fc2_1279_gridsearch_results.txt'
    #         file_name = f'model.decoder.layers.{i}.final_layer_norm_1279_gridsearch_results.txt'
    #     else:
    #         file_name = f'slurm_log_{i}.txt'
    #     with open(log_dir_2 + file_name, 'r') as f:
    #         a = f.readlines()
    #         for ia in range(len(a)):
    #             if 'model' in a[ia] and 'Functions to be tested' not in a[ia]:
    #                 for k in range(neuron):
    #                     _a = [j for j in a[ia].split()]
    #                     try:
    #                         if len(_a) == 12:
    #                             lat_sig[(i+layer-32) % n, (i+layer-32) // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), (i+layer-32) // n, float(_a[0].split('_')[-1].rstrip(':'))]
    #                         else:
    #                             lat_sig[(i+layer-32) % n, (i+layer-32) // n, k] = [float(_a[3][:-1]), 0, float(_a[6][:-1]), float(_a[-1]), (i+layer-32) // n, float(_a[0].split('_')[-1].rstrip(':'))]
    #                     except: 
    #                         pass
    #                     ia += 1
    #                 break
    
    for i in range(layer):
        # file_name = f'slurm_log_sensor_{i}.txt'
        file_name = f'slurm_log_{i}.txt'
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

    print(lat_sig[0, 0])
    print(lat_sig.shape)

    plt.figure(3)
    fig, ax = plt.subplots()

    stds = []

    lat_sig = lat_sig.reshape(lat_sig.shape[0], -1, lat_sig.shape[3])

    # import ipdb;ipdb.set_trace()

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

    # import ipdb;ipdb.set_trace()

    #lat_i = np.argmax(lat_sig[i, :, :, 3], axis=1)
    #print(lat_i)
    #print(lat_sig[i, 0])
    # _lats = np.array([lat_sig[i, j, lat_i[j], :] for j in range(lat_sig.shape[1]) if lat_sig[i, j, lat_i[j], 0] != 0])

    # import ipdb;ipdb.set_trace()
    _lats = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])
    # _lats : (point, (latency, corr, sensor, -log(pval), layer, neuron))
    stds.append(np.std(_lats[:, 0]))

    if exclude_tvl:
        lat_sig = np.zeros((n, layer, neuron, 6))
        for i in range(layer):
            # file_name = f'slurm_log_sensor_{i}.txt'
            file_name = f'slurm_log_{i}.txt'
            with open(log_tvl_dir + file_name, 'r') as f:
                a = f.readlines()
                a = [line for line in a if 'Time' not in line]
                for ia in range(len(a)):
                    if 'layer' in a[ia] and 'Functions to be tested' not in a[ia] and 'Transforms to be tested' not in a[ia]:
                        for k in range(neuron):
                            _a = [j for j in a[ia].split()]
                            try:
                                lat_sig[i % n, i // n, k] = [float(_a[3][:-1]), float(_a[6]), float(_a[9][:-1]), float(_a[11]), i // n, float(_a[0].split('_')[-1].rstrip(':'))]
                            except:
                                pass
                            ia += 1
                        break        
        lat_sig = lat_sig.reshape(lat_sig.shape[0], -1, lat_sig.shape[3])
        _lats_tvl = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)])

        mask = np.array([i for i in range(_lats.shape[0]) if np.any(np.all(_lats[i, 4:] == _lats_tvl[:, 4:], axis=1))])
        mask_non_tvl = np.array([i for i in range(_lats.shape[0]) if not np.any(np.all(_lats[i, 4:] == _lats_tvl[:, 4:], axis=1))])
        # import ipdb;ipdb.set_trace()

    if exclude_tvl:
        scatter = ax.scatter(_lats[mask_non_tvl, 0], _lats[mask_non_tvl, 4], c= _lats[mask_non_tvl, 4], cmap='brg', marker='.', s=15)
        scatter = ax.scatter(_lats[mask, 0], _lats[mask, 4], c='black', marker='.', s=4, alpha=0.6)
    else:
        scatter = ax.scatter(_lats[:, 0], _lats[:, 4], c= _lats[:, 4], cmap='brg', marker='.', s=15)
        cbar = plt.colorbar(scatter, ax=ax, label='layers')
    ax.set_ylim(-1, layer)
    # ax.scatter(lat_sig[i, :1, 0], lat_sig[i, :1, 3], marker='o')
    #for j in range(_lats.shape[0]):
    #    ax.annotate(j+1, (_lats[j, 0], _lats[j, 3]))

    plt.ylabel('Layer number')
    plt.xlabel('Latencies (ms)')
    plt.title(f'Threshold -log(p-value): {thres}')
    plt.xlim(-200, x_upper)
    # plt.legend()
    # plt.xlim(-10, 60)
    # plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/scatter/{size}_{neuron_selection}_select_excl_tvl', dpi=600)
    # plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/salmonn_omni/sensor/scatter/{neuron_selection}_20', dpi=600)
    plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_word/new_layer', dpi=600)

if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()