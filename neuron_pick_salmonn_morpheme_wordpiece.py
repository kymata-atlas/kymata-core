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

def process_lat_sig(n, log_dir, layer, neuron, thres, neuron_selection):

    lat_sig = read_log_file_asr(n, log_dir, layer, neuron)

    _lats = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres)])
    # _lats : (point, (latency, corr, sensor, -log(pval), layer, neuron))

    selected = selection(_lats, neuron_selection, layer)

    print(selected.shape[0])

    return selected

def asr_models_loop_full():

    layer = 33 # 66 64 34
    neuron = 4096
    neuron_selection = 'layer_sep'
    n = 1

    log_dir_morpheme = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_morpheme/log/'
    log_dir_wordpiece = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_wordpiece/log/'
    log_dir_phone = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_phone/log/'
    log_dir_word = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_word/log/'
    log_dir_morpheme_tvl = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_morpheme/tvl/log/'
    log_dir_wordpiece_tvl = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_wordpiece/tvl/log/'
    log_dir_phone_tvl = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_phone/tvl/log/'
    log_dir_word_tvl = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_word/tvl/log/'

    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*370*neuron*layer)))))
    thres_tvl = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*11*neuron*layer)))))

    selected_morpheme = process_lat_sig(n, log_dir_morpheme, layer, neuron, thres, neuron_selection)
    selected_wordpiece = process_lat_sig(n, log_dir_wordpiece, layer, neuron, thres, neuron_selection)
    selected_phone = process_lat_sig(n, log_dir_phone, layer, neuron, thres, neuron_selection)
    selected_word = process_lat_sig(n, log_dir_word, layer, neuron, thres, neuron_selection)

    # import ipdb;ipdb.set_trace()

    all_arrays = [selected_morpheme, selected_wordpiece, selected_phone, selected_word]

    # Track origin of each row by tagging it with its source index
    tagged_rows = []
    for source_idx, arr in enumerate(all_arrays):
        for row in arr:
            key = tuple(row[4:])  # based on last two columns
            val_3 = row[3]
            tagged_rows.append((key, val_3, row, source_idx))

    # Create a dictionary to store the best row for each key
    row_dict = {}
    for key, val_3, row, source_idx in tagged_rows:
        if key not in row_dict or val_3 > row_dict[key][1]:
            row_dict[key] = (row, val_3, source_idx)

    # Collect the filtered rows into their respective arrays
    arr1_new, arr2_new, arr3_new, arr4_new = [], [], [], []

    for row, _, source_idx in row_dict.values():
        if source_idx == 0:
            arr1_new.append(row)
        elif source_idx == 1:
            arr2_new.append(row)
        elif source_idx == 2:
            arr3_new.append(row)
        elif source_idx == 3:
            arr4_new.append(row)

    # Convert lists to numpy arrays
    morpheme_neurons = np.array(arr1_new)
    wordpiece_neurons = np.array(arr2_new)
    phone_neurons = np.array(arr3_new)
    word_neurons = np.array(arr4_new)

    # (peak lat, peak corr, ind, -log(pval), layer_no, neuron_no)

    lat_sig = read_log_file_asr(n, log_dir_morpheme_tvl, layer, neuron)
    morpheme_neurons_tvl = [lat_sig[0, j, 4:] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)]
    neuron_picks_morpheme = [
        [int(neuron[4]), int(neuron[5])] 
        for neuron in morpheme_neurons 
        if not any(np.array_equal(np.array(neuron[4:]), tvl) for tvl in morpheme_neurons_tvl)
    ]

    lat_sig = read_log_file_asr(n, log_dir_wordpiece_tvl, layer, neuron)
    wordpiece_neurons_tvl = [lat_sig[0, j, 4:] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)]
    neuron_picks_wordpiece = [
        [int(neuron[4]), int(neuron[5])] 
        for neuron in wordpiece_neurons 
        if not any(np.array_equal(np.array(neuron[4:]), tvl) for tvl in wordpiece_neurons_tvl)
    ]

    lat_sig = read_log_file_asr(n, log_dir_phone_tvl, layer, neuron)
    phone_neurons_tvl = [lat_sig[0, j, 4:] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)]
    neuron_picks_phone = [
        [int(neuron[4]), int(neuron[5])] 
        for neuron in phone_neurons 
        if not any(np.array_equal(np.array(neuron[4:]), tvl) for tvl in phone_neurons_tvl)
    ]
    lat_sig = read_log_file_asr(n, log_dir_word_tvl, layer, neuron)
    word_neurons_tvl = [lat_sig[0, j, 4:] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)]
    neuron_picks_word = [
        [int(neuron[4]), int(neuron[5])] 
        for neuron in word_neurons 
        if not any(np.array_equal(np.array(neuron[4:]), tvl) for tvl in word_neurons_tvl)
    ]

    neuron_picks_phone_old = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/phone_sig.npy').tolist()
    neuron_picks_phone_old += np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/other_phone_sig.npy').tolist()

    neuron_picks_word_old = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/word_sig.npy').tolist()
    neuron_picks_word_old += np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/other_word_sig.npy').tolist()

    neuron_picks_morpheme_old = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/morpheme_all.npy').tolist()
    neuron_picks_wordpiece_old = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/wordpiece_all.npy').tolist()

    import ipdb; ipdb.set_trace()

    phone_new = np.array([i for i in neuron_picks_phone if i not in neuron_picks_phone_old])
    word_new = np.array([i for i in neuron_picks_word if i not in neuron_picks_word_old])
    morpheme_new = np.array([i for i in neuron_picks_morpheme if i not in neuron_picks_morpheme_old])
    wordpiece_new = np.array([i for i in neuron_picks_wordpiece if i not in neuron_picks_wordpiece_old])

    print(phone_new.shape)
    print(word_new.shape)
    print(morpheme_new.shape)
    print(wordpiece_new.shape)

    np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/new_batch/phone.npy', phone_new)
    np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/new_batch/word.npy', word_new)
    np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/new_batch/morpheme.npy', morpheme_new)
    np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/new_batch/wordpiece.npy', wordpiece_new)

    print(len(neuron_picks_morpheme))
    print(len(neuron_picks_wordpiece))
    print(len(neuron_picks_phone))
    print(len(neuron_picks_word))

    # np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/all_cat/morpheme_all.npy', np.array(neuron_picks_morpheme))
    # np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/all_cat/wordpiece_all.npy', np.array(neuron_picks_wordpiece))
    # np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/all_cat/phone_all.npy', np.array(neuron_picks_phone))
    # np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/all_cat/word_all.npy', np.array(neuron_picks_word))

if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()