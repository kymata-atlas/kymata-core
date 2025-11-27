from logging import basicConfig, INFO
from pathlib import Path
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist

from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set
from kymata.plot.expression import expression_plot, legend_display_dict
from kymata.plot.color import constant_color_dict, gradient_color_dict

import time
from tqdm import tqdm

colour_map = {'word': 'red', 'phone': 'green', 'morpheme': 'blue', 'wordpiece': 'orange'}

def load_all_expression_data(base_folder):
    expression_data = None
    # Loop through each subdirectory inside the base folder
    for subdir in os.listdir(base_folder):
        subdir_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subdir_path):  # Ensure we are processing directories
            # List all .nkg files inside the subdirectory
            nkg_files = [f for f in os.listdir(subdir_path) if f.endswith('.nkg')]
            for nkg_file in nkg_files:
                file_path = os.path.join(subdir_path, nkg_file)
                if expression_data is None:
                    # Load the first .nkg file
                    expression_data = load_expression_set(file_path)
                else:
                    # Add data from subsequent .nkg files
                    expression_data += load_expression_set(file_path)
    return expression_data

def load_part_of_expression_data(base_folder, pick):
    expression_data = None
    # Loop through each subdirectory inside the base folder
    for subdir in os.listdir(base_folder):
        subdir_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subdir_path) and [int(x) for x in subdir.split('_')] in pick.tolist():  # Ensure we are processing directories
            # List all .nkg files inside the subdirectory
            nkg_files = [f for f in os.listdir(subdir_path) if f.endswith('.nkg')]
            for nkg_file in nkg_files:
                file_path = os.path.join(subdir_path, nkg_file)
                if expression_data is None:
                    # Load the first .nkg file
                    expression_data = load_expression_set(file_path)
                else:
                    # Add data from subsequent .nkg files
                    expression_data += load_expression_set(file_path)
    return expression_data

def plot_scatter(best_transforms, level, thres, ax):

    for i in tqdm(best_transforms[0], desc='Left hemisphere'):
        name = i.transform
        p_value = -i.logp_value
        latency = i.latency * 1000
        if p_value > thres and level in name:
            scatter = ax.scatter(latency, int(name.split('_')[0].replace('layer', '')), c= colour_map[level], marker='.', s=15)

    for i in tqdm(best_transforms[1], desc='Right hemisphere'):
        name = i.transform
        p_value = -i.logp_value
        latency = i.latency * 1000
        if p_value > thres and level in name:
            scatter = ax.scatter(latency, int(name.split('_')[0].replace('layer', '')), c= colour_map[level], marker='.', s=15)

def kill_neurons(best_transforms, level, thres):

    new_transforms = [[i.latency * 1000, int(i.transform.split('_')[0].replace('layer', '')), i.transform.split('_')[1], -i.logp_value] for i in best_transforms[0] if level in i.transform] # [[lat, layer, neuron, -logp]]
    new_transforms += [[i.latency * 1000, int(i.transform.split('_')[0].replace('layer', '')), i.transform.split('_')[1], -i.logp_value] for i in best_transforms[1] if level in i.transform] # [[lat, layer, neuron, -logp]]

    # Group by neuron and keep only the element with the smallest layer number
    # If layer numbers are the same, keep the one with the highest logp
    neuron_dict = {}
    for item in new_transforms:
        lat, layer, neuron, logp = item
        if logp > thres:
            # if neuron not in neuron_dict or (layer == neuron_dict[neuron][1] and logp > neuron_dict[neuron][3]) or layer < neuron_dict[neuron][1]:
            #     neuron_dict[neuron] = item
            if f'{layer}_{neuron}' not in neuron_dict or logp > neuron_dict[f'{layer}_{neuron}'][3]:
                neuron_dict[f'{layer}_{neuron}'] = item

    new_transforms = list(neuron_dict.values())

    return new_transforms


def plot_scatter_selected(best_transforms, thres, ax, level):

    out_dir = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/new_exp"
    out_file = os.path.join(out_dir, "selected_neurons.txt")
    with open(out_file, "a") as f:

        for i in tqdm(best_transforms, desc='Selected neurons'):
            latency = i[0]
            layer = i[1]
            neuron = i[2]
            p_value = i[3]
            scatter = ax.scatter(latency, layer, c= colour_map[level], marker='.', s=15)
            f.write(f'Neuron: {layer}_{neuron}, {level}, Latency: {latency}, -log(p-value): {p_value}\n')

def main():

    start_time = time.time()

    neuron_selection = False

    reduce_duplicates = False
    
    transform_family_type = 'standard'
    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-core", "kymata-core-data", "output")

    if transform_family_type == 'standard':

        expression_data_tvl = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/english_TVL_family_source_baseline.nkg')
        # expression_data_phone = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/source_nkg/phone.nkg')
        # expression_data_word = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/source_nkg/word.nkg')
        # expression_data_morpheme = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/source_nkg/morpheme.nkg')
        # expression_data_wordpiece = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/source_nkg/wordpiece.nkg')
        expression_data_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron_phone')
        expression_data_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron')
        expression_data_morpheme = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron_morpheme')
        # expression_data_wordpiece = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/source_nkg/wordpiece.nkg')
        phone_name = expression_data_phone.transforms
        word_name = expression_data_word.transforms
        morpheme_name = expression_data_morpheme.transforms
        # wordpiece_name = expression_data_wordpiece.transforms
        tvl_name = expression_data_tvl.transforms
        IL_name = [i for i in tvl_name if i != 'STL']
        STL_name = ['STL']

        for i in range(len(phone_name)):
            expression_data_phone.rename({phone_name[i]:f'{phone_name[i]}_phone'})
        phone_name = expression_data_phone.transforms

        for i in range(len(word_name)):
            expression_data_word.rename({word_name[i]:f'{word_name[i]}_word'})
        word_name = expression_data_word.transforms        

        for i in range(len(morpheme_name)):
            expression_data_morpheme.rename({morpheme_name[i]:f'{morpheme_name[i]}_morpheme'})
        morpheme_name = expression_data_morpheme.transforms

        # for i in range(len(wordpiece_name)):
        #     expression_data_wordpiece.rename({wordpiece_name[i]:f'{wordpiece_name[i]}_wordpiece'})
        # wordpiece_name = expression_data_wordpiece.transforms

        all_name = word_name + IL_name + STL_name + phone_name + morpheme_name # + wordpiece_name

        expression_data_all = expression_data_word[[i for i in word_name if '2298' not in i]] + expression_data_tvl + expression_data_phone + expression_data_morpheme # + expression_data_wordpiece

        # best_transforms = expression_data_all.best_transforms()

        # alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
        # thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (2*200*370*4096*33*3)))))

        # neuron_dict = {}

        # for i in best_transforms[0]:
        #     name = i.transform
        #     p_value = -i.logp_value
        #     latency = i.latency * 1000
        #     if p_value > thres and 'word' in name and (name not in neuron_dict or p_value > neuron_dict[name][1]):
        #         neuron_dict[name] = (latency, p_value)

        # for i in best_transforms[1]:
        #     name = i.transform
        #     p_value = -i.logp_value
        #     latency = i.latency * 1000
        #     if p_value > thres and 'word' in name and (name not in neuron_dict or p_value > neuron_dict[name][1]):
        #         neuron_dict[name] = (latency, p_value)

        # neuron_dict = dict(sorted(neuron_dict.items()))
        # out_file_neuron_dict = os.path.join("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/new_exp", "neuron_dict_no_select.txt")
        # with open(out_file_neuron_dict, "w") as f:
        #     for name, (latency, p_value) in neuron_dict.items():
        #         f.write(f'{name}: Latency: {latency}, -log(p-value): {p_value}\n')



        # fig = expression_plot(expression_data_all, paired_axes=True, minimap=None, show_legend=True, show_only=['layer18_23_word'], hidden_transforms_in_legend=False,
        #                         color=constant_color_dict(word_name, color= 'red')
        #                             | constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink'))
        
        # fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/new_exp/word_name_exp_select_test.png")

        # fig = expression_plot(expression_data_all, paired_axes=True, minimap='large', show_legend=True, show_only=word_name, hidden_transforms_in_legend=False,
        #                         color=constant_color_dict(word_name, color= 'red')
        #                             | constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink')
        #                             | constant_color_dict(phone_name, color='green')
        #                             | constant_color_dict(morpheme_name, color='blue'),
        #                             # | constant_color_dict(wordpiece_name, color='orange'),
        #                         legend_display=legend_display_dict(word_name, 'SALMONN word features')
        #                             | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness transform')
        #                             | legend_display_dict(phone_name, 'SALMONN phone features')
        #                             | legend_display_dict(morpheme_name, 'SALMONN morpheme features'))
        #                             # | legend_display_dict(wordpiece_name, 'SALMONN wordpiece features'))
        # fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/no_2298/word_only.png")
        # plt.close(fig)

        # fig = expression_plot(expression_data_all, paired_axes=True, minimap='large', show_legend=True, show_only=tvl_name, hidden_transforms_in_legend=False,
        #                         color=constant_color_dict(word_name, color= 'red')
        #                             | constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink')
        #                             | constant_color_dict(phone_name, color='green')
        #                             | constant_color_dict(morpheme_name, color='blue'),
        #                             # | constant_color_dict(wordpiece_name, color='orange'),
        #                         legend_display=legend_display_dict(word_name, 'SALMONN word features')
        #                             | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness transform')
        #                             | legend_display_dict(phone_name, 'SALMONN phone features')
        #                             | legend_display_dict(morpheme_name, 'SALMONN morpheme features'))
        #                             # | legend_display_dict(wordpiece_name, 'SALMONN wordpiece features'))
        # fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/no_2298/tvl_only.png")
        # plt.close(fig)

        # fig = expression_plot(expression_data_all, paired_axes=True, minimap='large', show_legend=True, show_only=phone_name, hidden_transforms_in_legend=False,
        #                         color=constant_color_dict(word_name, color= 'red')
        #                             | constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink')
        #                             | constant_color_dict(phone_name, color='green')
        #                             | constant_color_dict(morpheme_name, color='blue'),
        #                             # | constant_color_dict(wordpiece_name, color='orange'),
        #                         legend_display=legend_display_dict(word_name, 'SALMONN word features')
        #                             | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness transform')
        #                             | legend_display_dict(phone_name, 'SALMONN phone features')
        #                             | legend_display_dict(morpheme_name, 'SALMONN morpheme features'))
        #                             # | legend_display_dict(wordpiece_name, 'SALMONN wordpiece features'))
        # fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/no_2298/phone_only.png")
        # plt.close(fig)

        # fig = expression_plot(expression_data_all, paired_axes=True, minimap='large', show_legend=True, show_only=morpheme_name, hidden_transforms_in_legend=False,
        #                         color=constant_color_dict(word_name, color= 'red')
        #                             | constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink')
        #                             | constant_color_dict(phone_name, color='green')
        #                             | constant_color_dict(morpheme_name, color='blue'),
        #                             # | constant_color_dict(wordpiece_name, color='orange'),
        #                         legend_display=legend_display_dict(word_name, 'SALMONN word features')
        #                             | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness transform')
        #                             | legend_display_dict(phone_name, 'SALMONN phone features')
        #                             | legend_display_dict(morpheme_name, 'SALMONN morpheme features'))
        #                             # | legend_display_dict(wordpiece_name, 'SALMONN wordpiece features'))
        # fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/no_2298/morpheme_only.png")
        # plt.close(fig)

        # fig = expression_plot(expression_data_all, paired_axes=True, minimap='large', show_legend=True, hidden_transforms_in_legend=False,
        #                         color=constant_color_dict(word_name, color= 'red')
        #                             | constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink')
        #                             | constant_color_dict(phone_name, color='green')
        #                             | constant_color_dict(morpheme_name, color='blue'),
        #                             # | constant_color_dict(wordpiece_name, color='orange'),
        #                         legend_display=legend_display_dict(word_name, 'SALMONN word features')
        #                             | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness transform')
        #                             | legend_display_dict(phone_name, 'SALMONN phone features')
        #                             | legend_display_dict(morpheme_name, 'SALMONN morpheme features'))
        #                             # | legend_display_dict(wordpiece_name, 'SALMONN wordpiece features'))
        # fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/no_2298/all_together.png")
        # plt.close(fig)

        for i in range(1):

            fig = expression_plot(expression_data_all, paired_axes=True, minimap='large', show_legend=True, show_only=[name for name in all_name if f'layer{i}_' in name],
            # fig = expression_plot(expression_data_all[[name for name in all_name if f'layer{i}_' in name or name in IL_name + STL_name]], paired_axes=True, minimap='large', show_legend=True, show_only=[name for name in all_name if f'layer{i}_' in name],
                                    color=constant_color_dict(word_name, color= 'red')
                                        | constant_color_dict(IL_name, color= 'purple')
                                        | constant_color_dict(STL_name, color= 'pink')
                                        | constant_color_dict(phone_name, color='green')
                                        | constant_color_dict(morpheme_name, color='blue'),
                                        # | constant_color_dict(wordpiece_name, color='orange'),
                                    legend_display=legend_display_dict(word_name, 'SALMONN word features')
                                        | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
                                        | legend_display_dict(STL_name, 'Short Term Loudness transform')
                                        | legend_display_dict(phone_name, 'SALMONN phone features')
                                        | legend_display_dict(morpheme_name, 'SALMONN morpheme features'))
                                        # | legend_display_dict(wordpiece_name, 'SALMONN wordpiece features'))

            # fig.savefig("/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output/music/20_participants/all_segments/music.png")
            # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/last_four_reps_tvl/last_four_reps_source.png")
            fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/no_2298/layer/layer_{i}_brain_without_wordpiece.png")
            plt.close(fig)

        # fig = expression_plot(expression_data_all, paired_axes=True, minimap='large', show_legend=True, show_only=tvl_name,
        # # fig = expression_plot(expression_data_all[[name for name in all_name if f'layer{i}_' in name or name in IL_name + STL_name]], paired_axes=True, minimap='large', show_legend=True, show_only=[name for name in all_name if f'layer{i}_' in name],
        #                         color=constant_color_dict(word_name, color= 'red')
        #                             | constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink')
        #                             | constant_color_dict(phone_name, color='green')
        #                             | constant_color_dict(morpheme_name, color='blue'),
        #                             # | constant_color_dict(wordpiece_name, color='orange'),
        #                         legend_display=legend_display_dict(word_name, 'SALMONN word features')
        #                             | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness transform')
        #                             | legend_display_dict(phone_name, 'SALMONN phone features')
        #                             | legend_display_dict(morpheme_name, 'SALMONN morpheme features'))
        #                             # | legend_display_dict(wordpiece_name, 'SALMONN wordpiece features'))

        # # fig.savefig("/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output/music/20_participants/all_segments/music.png")
        # # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/last_four_reps_tvl/last_four_reps_source.png")
        # fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/no_2298/layer/layer_0_brain_without_wordpiece.png")
        # plt.close(fig)
        




        # best_transforms = expression_data_all.best_transforms()

        # # import ipdb; ipdb.set_trace()

        # plt.figure(3)
        # fig, ax = plt.subplots()

        # alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
        # thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (2*200*370*4096*33*3)))))

        # if not reduce_duplicates:

        #     plot_scatter(best_transforms, 'word', thres, ax)
        #     plot_scatter(best_transforms, 'phone', thres, ax)
        #     plot_scatter(best_transforms, 'morpheme', thres, ax)
        #     # plot_scatter(best_transforms, 'wordpiece', thres, ax)
        
        # else:

        #     best_transforms_word = kill_neurons(best_transforms, 'word', thres)
        #     plot_scatter_selected(best_transforms_word, thres, ax, 'word')
        #     best_transforms_phone = kill_neurons(best_transforms, 'phone', thres)
        #     plot_scatter_selected(best_transforms_phone, thres, ax, 'phone')
        #     best_transforms_morpheme = kill_neurons(best_transforms, 'morpheme', thres)
        #     plot_scatter_selected(best_transforms_morpheme, thres, ax, 'morpheme')
        #     best_transforms_wordpiece = kill_neurons(best_transforms, 'wordpiece', thres)
        #     plot_scatter_selected(best_transforms_wordpiece, thres, ax, 'wordpiece')

        # ax.set_ylim(-1, 33)

        # plt.ylabel('Layer number')
        # plt.xlabel('Latencies (ms)')
        # plt.title(f'Threshold -log(p-value): {thres}')
        # plt.xlim(-200, 800)

        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/new_exp/word_high_latencies.png")

        # plt.close(fig)

        elapsed_time = time.time() - start_time
        print(f"Script completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == '__main__':
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    main()
