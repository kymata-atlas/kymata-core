import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist
from kymata.io.nkg import load_expression_set

phone_dict = {0: 'Consonantal',
              1: 'Sonorant',
              2: 'Voiced',
              3: 'Nasal',
              4: 'Plosive',
              5: 'Fricative',
              6: 'Approximant',
              7: 'Labial',
              8: 'Coronal',
              9: 'Dorsal',
              10: 'High',
              11: 'Mid',
              12: 'Low',
              13: 'Front',
              14: 'Central',
              15: 'Back',
              16: 'Round',
              17: 'Tense',
              18: 'AA',
              19: 'AE',
              20: 'AH',
              21: 'AO',
              22: 'AW',
              23: 'AY',
              24: 'B',
              25: 'CH',
              26: 'D',
              27: 'DH',
              28: 'EH',
              29: 'ER',
              30: 'EY',
              31: 'F',
              32: 'G',
              33: 'HH',
              34: 'IH',
              35: 'IY',
              36: 'JH',
              37: 'K',
              38: 'L',
              39: 'M',
              40: 'N',
              41: 'NG',
              42: 'OW',
              43: 'OY',
              44: 'P',
              45: 'R',
              46: 'S',
              47: 'SH',
              48: 'T',
              49: 'TH',
              50: 'UH',
              51: 'UW',
              52: 'V',
              53: 'W',
              54: 'Y',
              55: 'Z',
              56: 'ZH'}

word_dict = {0: 'Log Frequency',
             1: 'Number of Phonological Neighbours',
             2: 'Frequency of Phonological Neighbours',
             3: 'Correctness Rating',
             4: 'Semantic Neighborhood Density',
             5: 'Semantic Diversity',
             6: 'Age Of Acquisition',
             7: 'Body Object Interaction',
             8: 'Emotional Valence',
             9: 'Emotional Arousal',
             10: 'Emotional Dominance',
             11: 'Mean Reaction Time (Lexical Decision)',
             12: 'Mean Reaction Time (Naming)',
             13: 'Occurrences',
             14: 'Preposition',
             15: 'Number',
             16: 'Noun',
             17: 'Determiner',
             18: 'Conjunction',
             19: 'Adverb',
             20: 'Adjective',
             21: 'Verb',
             22: 'Pronoun',
             23: 'Number of Open Nodes',
             24: 'Number of Close Nodes',
             25: 'Sentence End',
             26: 'Tree Depth',
             27: 'Linear Order'
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
    figure_opt = 'cat_phone_1'
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

    lat_sig = read_log_file_asr(n, tvl_log_dir, layer, neuron)
    _lats_tvl = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)])
    lat_sig = read_log_file_asr(n, tvl_compare_log_dir, layer, neuron)
    _lats_tvl_base = np.array([lat_sig[0, j, :] for j in range(lat_sig.shape[1]) if (lat_sig[0, j, 0] != 0 and lat_sig[0, j, 3] > thres_tvl)])

    mask_phone_enhanced = np.array([i for i in range(enhanced.shape[0]) if not np.any(np.all(enhanced[i, 4:] == _lats_tvl[:, 4:], axis=1))])
    mask_phone_emerge = np.array([i for i in range(emerge.shape[0]) if not np.any(np.all(emerge[i, 4:] == _lats_tvl[:, 4:], axis=1))])
    mask_phone_reduced = np.array([i for i in range(reduced.shape[0]) if not np.any(np.all(reduced[i, 4:] == _lats_tvl_base[:, 4:], axis=1))])
    mask_phone_demolish = np.array([i for i in range(demolish.shape[0]) if not np.any(np.all(demolish[i, 4:] == _lats_tvl_base[:, 4:], axis=1))])


    if 'phone' in figure_opt:
        feats_path = f'/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_phone_pvalue.npy'
        feats = np.load(feats_path)
        counter = 0
        if figure_opt == 'cat_phone_1':
            counter_vector = np.zeros((57,))
            feats[6:, :, :] = np.ones((51, neuron, layer))
        elif figure_opt == 'cat_phone_2':
            counter_vector = np.zeros((57,))
            feats[:23, :, :] = np.ones((23, neuron, layer))
            feats[27:, :, :] = np.ones((30, neuron, layer))
        else:
            counter_vector = np.zeros((57,))
        mask_feats_1 = []
        mask_feats_2 = []
        for i in range(mask_phone_enhanced.shape[0]):
            dim = int(enhanced[mask_phone_enhanced[i], 5])
            lay = int(enhanced[mask_phone_enhanced[i], 4])
            if np.min(feats[:, dim, lay]) < thres_feats:
                print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with phonetic feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])} with latency {enhanced[mask_phone_enhanced[i], 0]}')
                counter += 1
                counter_vector[np.argmin(feats[:, dim, lay])] += 1
                mask_feats_1.append([mask_phone_enhanced[i], np.argmin(feats[:, dim, lay])])
        for i in range(mask_phone_emerge.shape[0]):
            dim = int(emerge[mask_phone_emerge[i], 5])
            lay = int(emerge[mask_phone_emerge[i], 4])
            if np.min(feats[:, dim, lay]) < thres_feats:
                print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with phonetic feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])} with latency {emerge[mask_phone_emerge[i], 0]}')
                counter += 1
                counter_vector[np.argmin(feats[:, dim, lay])] += 1
                mask_feats_2.append([mask_phone_emerge[i], np.argmin(feats[:, dim, lay])])
        print(f'Proportion of significant neurons: {counter/(mask_phone_enhanced.shape[0]+mask_phone_emerge.shape[0])}')
        print(counter_vector)
        mask_feats_1 = np.array(mask_feats_1)
        mask_feats_2 = np.array(mask_feats_2)
        feats_to_disp = [i for i, occur in enumerate(counter_vector) if occur > occur_thres]
        colours = generate_green_variations(6)
        for i, ind in enumerate(feats_to_disp):
            new_mask = [k for j, k in enumerate(mask_feats_1[:, 0]) if mask_feats_1[j, 1] == ind]
            if x_data == 'latency':
                scatter = ax.scatter(enhanced[new_mask, 0], enhanced[new_mask, 4], color=colours[i], marker='o', s=10, label = f'{phone_dict[ind]}')
            else:
                scatter = ax.scatter(enhanced[new_mask, 5], enhanced[new_mask, 4], color=colours[i], marker='o', s=10, label = f'{phone_dict[ind]}')
        for i, ind in enumerate(feats_to_disp):
            new_mask = [k for j, k in enumerate(mask_feats_2[:, 0]) if mask_feats_2[j, 1] == ind]
            if x_data == 'latency':
                scatter = ax.scatter(emerge[new_mask, 0], emerge[new_mask, 4], color=colours[i], marker='o', s=10)
            else:
                scatter = ax.scatter(emerge[new_mask, 5], emerge[new_mask, 4], color=colours[i], marker='o', s=10)

        if x_data == 'latency':
            scatter = ax.scatter(enhanced[np.setdiff1d(mask_phone_enhanced, mask_feats_1), 0], enhanced[np.setdiff1d(mask_phone_enhanced, mask_feats_1), 4], color='green', marker='.', s=5, alpha= 0.15, label = 'Other Phonetic Features')
            scatter = ax.scatter(emerge[np.setdiff1d(mask_phone_emerge, mask_feats_2), 0], emerge[np.setdiff1d(mask_phone_emerge, mask_feats_2), 4], color='green', marker='.', s=5, alpha= 0.15)
        else:
            scatter = ax.scatter(enhanced[np.setdiff1d(mask_phone_enhanced, mask_feats_1), 5], enhanced[np.setdiff1d(mask_phone_enhanced, mask_feats_1), 4], color='green', marker='.', s=5, alpha= 0.15, label = 'Other Phonetic Features')
            scatter = ax.scatter(emerge[np.setdiff1d(mask_phone_emerge, mask_feats_2), 5], emerge[np.setdiff1d(mask_phone_emerge, mask_feats_2), 4], color='green', marker='.', s=5, alpha= 0.15)            


    else:
        feats_path = f'/imaging/projects/cbu/kymata/analyses/tianyi/workspace/output/corr/salmonn_7B_word_with_class_syntax_pvalue.npy'
        feats = np.load(feats_path)
        feats[6:14, :, :] = np.ones((8, neuron, layer))

        counter = 0
        if figure_opt == 'cat_pos':
            counter_vector = np.zeros((28,))
            feats[:14, :, :] = np.ones((14, neuron, layer))
            feats[23:, :, :] = np.ones((5, neuron, layer))
        elif figure_opt == 'cat_lex':
            counter_vector = np.zeros((28,))
            feats[3:, :, :] = np.ones((25, neuron, layer))
        elif figure_opt == 'cat_sem':
            counter_vector = np.zeros((28,))
            feats[:3, :, :] = np.ones((3, neuron, layer))
            feats[6:, :, :] = np.ones((22, neuron, layer))   
        elif figure_opt == 'cat_syn':
            counter_vector = np.zeros((28,))
            feats[:23, :, :] = np.ones((23, neuron, layer))
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
                counter += 1
                counter_vector[np.argmin(feats[:, dim, lay])] += 1
                mask_feats_1.append([mask_phone_reduced[i], np.argmin(feats[:, dim, lay])])
        for i in range(mask_phone_demolish.shape[0]):
            dim = int(demolish[mask_phone_demolish[i], 5])
            lay = int(demolish[mask_phone_demolish[i], 4])
            if np.min(feats[:, dim, lay]) < thres_feats:
                print(f'The Salmonn neuron {dim} at layer {lay} has the most significant correlation with word feature {np.argmin(feats[:, dim, lay])} with a p-value of {np.min(feats[:, dim, lay])}')
                counter += 1
                counter_vector[np.argmin(feats[:, dim, lay])] += 1
                mask_feats_2.append([mask_phone_demolish[i], np.argmin(feats[:, dim, lay])])
        print(f'Proportion of significant neurons: {counter/(mask_phone_reduced.shape[0]+mask_phone_demolish.shape[0])}')
        print(counter_vector)
        mask_feats_1 = np.array(mask_feats_1)
        mask_feats_2 = np.array(mask_feats_2)
        feats_to_disp = [i for i, occur in enumerate(counter_vector) if occur > occur_thres]
        colours = generate_red_variations(5)
        for i, ind in enumerate(feats_to_disp):
            new_mask = [k for j, k in enumerate(mask_feats_1[:, 0]) if mask_feats_1[j, 1] == ind]
            if x_data =='latency':
                scatter = ax.scatter(reduced[new_mask, 0], reduced[new_mask, 4], color=colours[i], marker='s', s=10, label = f'{word_dict[ind]}')
            else:
                scatter = ax.scatter(reduced[new_mask, 5], reduced[new_mask, 4], color=colours[i], marker='s', s=10, label = f'{word_dict[ind]}')
        for i, ind in enumerate(feats_to_disp):
            new_mask = [k for j, k in enumerate(mask_feats_2[:, 0]) if mask_feats_2[j, 1] == ind]
            if x_data =='latency':
                scatter = ax.scatter(demolish[new_mask, 0], demolish[new_mask, 4], color=colours[i], marker='s', s=10)
            else:
                scatter = ax.scatter(demolish[new_mask, 5], demolish[new_mask, 4], color=colours[i], marker='s', s=10)

        if x_data =='latency':
            scatter = ax.scatter(reduced[np.setdiff1d(mask_phone_reduced, mask_feats_1), 0], reduced[np.setdiff1d(mask_phone_reduced, mask_feats_1), 4], color='red', marker='.', s=5, alpha= 0.15, label = 'Other Word Features')
            scatter = ax.scatter(demolish[np.setdiff1d(mask_phone_demolish, mask_feats_2), 0], demolish[np.setdiff1d(mask_phone_demolish, mask_feats_2), 4], color='red', marker='.', s=5, alpha= 0.15)
        else:
            scatter = ax.scatter(reduced[np.setdiff1d(mask_phone_reduced, mask_feats_1), 5], reduced[np.setdiff1d(mask_phone_reduced, mask_feats_1), 4], color='red', marker='.', s=5, alpha= 0.15, label = 'Other Word Features')
            scatter = ax.scatter(demolish[np.setdiff1d(mask_phone_demolish, mask_feats_2), 5], demolish[np.setdiff1d(mask_phone_demolish, mask_feats_2), 4], color='red', marker='.', s=5, alpha= 0.15)           

        
    if x_data == 'latency':
        plt.xlabel('Latency (ms) relative to onset of the environment')
        ax.axvline(x=0, color='k', linestyle='dotted')
    else:
        plt.xlabel('Neuron number')
        x_upper = neuron + 200

    handles, labels = plt.gca().get_legend_handles_labels()

    # Create a dictionary to remove duplicates (preserve the order)
    unique_labels = dict(zip(labels, handles))

    # Create the legend with unique labels
    ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=5)

    ax.set_ylim(-1, layer)
    # ax.set_ylim(-100, neuron+100)
    # ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=5)

    # plt.ylabel('Salmonn neuron number')
    plt.ylabel('Salmonn layer number')

    # # Define y-tick positions and labels
    # yticks = [0, 5, 10, 15, 20, 25, 30]
    # ytick_labels = ['TVL-related functions', 5, 10, 15, 20, 25, 30]

    # # Set y-ticks and custom labels
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(ytick_labels)

    # plt.title(f'Threshold -log(p-value): {thres}')
    plt.xlim(-200, x_upper)
    plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/scatter/salmonn_7b_{figure_opt}_interpret_{thres_feats}_{occur_thres}_{x_data}_v5.png', dpi=600, bbox_inches="tight")


if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()