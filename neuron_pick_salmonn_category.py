import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist
from kymata.io.nkg import load_expression_set

phone_dict = {0: 'Articulatory Features',
              1: 'Articulatory Features',
              2: 'Articulatory Features',
              3: 'Articulatory Features',
              4: 'Articulatory Features',
              5: 'Articulatory Features',
              6: 'Articulatory Features',
              7: 'Articulatory Features',
              8: 'Articulatory Features',
              9: 'Articulatory Features',
              10: 'Articulatory Features',
              11: 'Articulatory Features',
              12: 'Articulatory Features',
              13: 'Articulatory Features',
              14: 'Articulatory Features',
              15: 'Articulatory Features',
              16: 'Articulatory Features',
              17: 'Articulatory Features',
              18: 'Phoneme Identities',
              19: 'Phoneme Identities',
              20: 'Phoneme Identities',
              21: 'Phoneme Identities',
              22: 'Phoneme Identities',
              23: 'Phoneme Identities',
              24: 'Phoneme Identities',
              25: 'Phoneme Identities',
              26: 'Phoneme Identities',
              27: 'Phoneme Identities',
              28: 'Phoneme Identities',
              29: 'Phoneme Identities',
              30: 'Phoneme Identities',
              31: 'Phoneme Identities',
              32: 'Phoneme Identities',
              33: 'Phoneme Identities',
              34: 'Phoneme Identities',
              35: 'Phoneme Identities',
              36: 'Phoneme Identities',
              37: 'Phoneme Identities',
              38: 'Phoneme Identities',
              39: 'Phoneme Identities',
              40: 'Phoneme Identities',
              41: 'Phoneme Identities',
              42: 'Phoneme Identities',
              43: 'Phoneme Identities',
              44: 'Phoneme Identities',
              45: 'Phoneme Identities',
              46: 'Phoneme Identities',
              47: 'Phoneme Identities',
              48: 'Phoneme Identities',
              49: 'Phoneme Identities',
              50: 'Phoneme Identities',
              51: 'Phoneme Identities',
              52: 'Phoneme Identities',
              53: 'Phoneme Identities',
              54: 'Phoneme Identities',
              55: 'Phoneme Identities',
              56: 'Phoneme Identities'}

word_dict = {0: 'Lexical Features',
             1: 'Lexical Features',
             2: 'Lexical Features',
             3: 'Semantic Features',
             4: 'Semantic Features',
             5: 'Semantic Features',
             6: 'Invalid',
             7: 'Invalid',
             8: 'Invalid',
             9: 'Invalid',
             10: 'Invalid',
             11: 'Semantic-Behavioural Features',
             12: 'Semantic-Behavioural Features',
             13: 'Invalid',
             14: 'Part of Speech',
             15: 'Part of Speech',
             16: 'Part of Speech',
             17: 'Part of Speech',
             18: 'Part of Speech',
             19: 'Part of Speech',
             20: 'Part of Speech',
             21: 'Part of Speech',
             22: 'Part of Speech',
             23: 'Syntactic Features',
             24: 'Syntactic Features',
             25: 'Syntactic Features',
             26: 'Syntactic Features',
             27: 'Syntactic Features'
             }

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
    x_upper = 800
    x_data = 'latency'
    neuron_selection = 'layer_sep'
    margin = 0
    n = 1
    figure_opt = 'phone'
    thres_feats = 0.001
    occur_thres = 0

    log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_7b_phone/fc2/log/'
    compare_log_dir = f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/salmonn_7b/fc2/log/'
    tvl_compare_log_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/salmonn_7b/tvl/log/'
    tvl_log_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_7b_phone/tvl/log/'

    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*370*neuron*layer))))) # maybe we should get rid of the 2 here because we don't split the hemispheres
    thres_tvl = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*11*neuron*layer)))))
    thres_tvl_true = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*11*370)))))

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

    lat_sig = read_log_file_asr(n, tvl_log_dir, layer, neuron)
    _lats_tvl = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)])
    lat_sig = read_log_file_asr(n, tvl_compare_log_dir, layer, neuron)
    _lats_tvl_base = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)])

    mask_phone_enhanced = np.array([i for i in range(enhanced.shape[0]) if not np.any(np.all(enhanced[i, 4:] == _lats_tvl[:, 4:], axis=1))])
    mask_phone_emerge = np.array([i for i in range(emerge.shape[0]) if not np.any(np.all(emerge[i, 4:] == _lats_tvl[:, 4:], axis=1))])
    mask_phone_reduced = np.array([i for i in range(reduced.shape[0]) if not np.any(np.all(reduced[i, 4:] == _lats_tvl_base[:, 4:], axis=1))])
    mask_phone_demolish = np.array([i for i in range(demolish.shape[0]) if not np.any(np.all(demolish[i, 4:] == _lats_tvl_base[:, 4:], axis=1))])

    neuron_picks = []

    neuron_picks_lex = []
    neuron_picks_sem = []
    neuron_picks_pos = []
    neuron_picks_syn = []
    neuron_pick_other = []

    neuron_picks_id = []
    neuron_picks_art = []

    if figure_opt == 'phone':
        feats_path = f'/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_{figure_opt}_pvalue.npy'
        feats = np.load(feats_path)
        counter = 0
        counter_vector = np.zeros((57,))
        mask_feats_1 = []
        mask_feats_2 = []
        for i in range(mask_phone_enhanced.shape[0]):
            dim = int(enhanced[mask_phone_enhanced[i], 5])
            lay = int(enhanced[mask_phone_enhanced[i], 4])
            if np.min(feats[:, dim, lay]) < thres_feats:
                print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with phonetic feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])} with latency {enhanced[mask_phone_enhanced[i], 0]}')
                neuron_picks.append([lay, dim, np.argmin(feats[:, dim, lay])])
                counter += 1
                counter_vector[np.argmin(feats[:, dim, lay])] += 1
                mask_feats_1.append([mask_phone_enhanced[i], np.argmin(feats[:, dim, lay])])
            else:
                neuron_pick_other.append([lay, dim])
        for i in range(mask_phone_emerge.shape[0]):
            dim = int(emerge[mask_phone_emerge[i], 5])
            lay = int(emerge[mask_phone_emerge[i], 4])
            if np.min(feats[:, dim, lay]) < thres_feats:
                print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with phonetic feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])} with latency {emerge[mask_phone_emerge[i], 0]}')
                neuron_picks.append([lay, dim, np.argmin(feats[:, dim, lay])])
                counter += 1
                counter_vector[np.argmin(feats[:, dim, lay])] += 1
                mask_feats_2.append([mask_phone_emerge[i], np.argmin(feats[:, dim, lay])])
            else:
                neuron_pick_other.append([lay, dim])
        print(f'Proportion of significant neurons: {counter/(mask_phone_enhanced.shape[0]+mask_phone_emerge.shape[0])}')
        print(counter_vector)
        print(counter)
        neuron_picks.sort(key=lambda x: x[-1])
        print(neuron_picks)
        np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/phone_detail.npy', neuron_picks)
 
    else:
        feats_path = f'/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_{figure_opt}_pvalue.npy'
        feats = np.load(feats_path)
        feats[6:14, :, :] = np.ones((8, neuron, layer))

        counter = 0
        if figure_opt == 'word':
            counter_vector = np.zeros((14,))
        elif figure_opt == 'word_with_class':
            counter_vector = np.zeros((23,))
        else:
            counter_vector = np.zeros((28,))
        mask_feats_1 = []
        mask_feats_2 = []
        for i in range(mask_phone_reduced.shape[0]):
            dim = int(reduced[mask_phone_reduced[i], 5])
            lay = int(reduced[mask_phone_reduced[i], 4])
            if np.min(feats[:, dim, lay]) < thres_feats:
                print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with word feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])}')
                neuron_picks.append([lay, dim, np.argmin(feats[:, dim, lay])])
                counter += 1
                counter_vector[np.argmin(feats[:, dim, lay])] += 1
                mask_feats_1.append([mask_phone_reduced[i], np.argmin(feats[:, dim, lay])])
            else:
                neuron_pick_other.append([lay, dim])
        for i in range(mask_phone_demolish.shape[0]):
            dim = int(demolish[mask_phone_demolish[i], 5])
            lay = int(demolish[mask_phone_demolish[i], 4])
            if np.min(feats[:, dim, lay]) < thres_feats:
                print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with word feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])}')
                neuron_picks.append([lay, dim, np.argmin(feats[:, dim, lay])])
                counter += 1
                counter_vector[np.argmin(feats[:, dim, lay])] += 1
                mask_feats_2.append([mask_phone_demolish[i], np.argmin(feats[:, dim, lay])])
            else:
                neuron_pick_other.append([lay, dim])
        print(f'Proportion of significant neurons: {counter/(mask_phone_reduced.shape[0]+mask_phone_demolish.shape[0])}')
        print(counter_vector)
        print(counter)
        neuron_picks.sort(key=lambda x: x[-1])
        print(neuron_picks)
        np.save('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/word_detail.npy', neuron_picks)


if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()