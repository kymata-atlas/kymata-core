from logging import basicConfig, INFO
from pathlib import Path
import os
from os import path
import numpy as np

from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set, save_expression_set
from kymata.plot.expression import expression_plot, legend_display_dict
from kymata.plot.color import constant_color_dict, gradient_color_dict

import time
from tqdm import tqdm

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

def main():

    transform_family_type = 'all_level_sensor'

    # template invoker for printing out expression set .nkgs

    if transform_family_type == 'all_level_source':

        expression_data_tvl = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/english_TVL_family_source_baseline_derangments_6.nkg')
        tvl_name = expression_data_tvl.transforms
        IL_name = [i for i in tvl_name if i != 'STL']
        STL_name = ['STL']

        # fig = expression_plot(expression_data_tvl, paired_axes=True, minimap='large', show_legend=True,
        #                         color=constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink'),
        #                         legend_display=legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness transform'))
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/test.png")

        expression_data_morpheme = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_morpheme_source')
        morpheme_list = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/all_cat/morpheme_all.npy')
        morpheme_name = [f'layer{i}_{j}' for i,j in morpheme_list]
        expression_data_wordpiece = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_wordpiece_source')
        wordpiece_list = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/all_cat/wordpiece_all.npy')
        wordpiece_name = [f'layer{i}_{j}' for i,j in wordpiece_list]
        expression_data_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_phone_source')
        expression_data_phone += load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron_phone')
        phone_list = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/all_cat/phone_all.npy')
        phone_name = [f'layer{i}_{j}' for i,j in phone_list]
        expression_data_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_word_source')
        expression_data_word += load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron')
        word_list = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/all_cat/word_all.npy')
        word_name = [f'layer{i}_{j}' for i,j in word_list]

        # save_expression_set(expression_data_morpheme[morpheme_name] + expression_data_wordpiece[wordpiece_name] + expression_data_phone[phone_name] + expression_data_word[word_name], 
        #                     '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/all_language_transforms.nkg')

        expression_data = expression_data_tvl + expression_data_morpheme[morpheme_name] + expression_data_wordpiece[wordpiece_name] + expression_data_phone[phone_name] + expression_data_word[word_name]


        fig = expression_plot(expression_data, paired_axes=True, minimap='large', show_legend=True, show_only=tvl_name,
        # fig = expression_plot(expression_data, paired_axes=True, minimap='large', show_legend=True,
                                color=constant_color_dict(word_name, color= 'red')
                                    | constant_color_dict(IL_name, color= 'purple')
                                    | constant_color_dict(STL_name, color= 'pink')
                                    | constant_color_dict(phone_name, color='green')
                                    | constant_color_dict(morpheme_name, color='blue')
                                    | constant_color_dict(wordpiece_name, color='orange'),
                                legend_display=legend_display_dict(word_name, 'SALMONN word features')
                                    | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
                                    | legend_display_dict(STL_name, 'Short Term Loudness transform')
                                    | legend_display_dict(phone_name, 'SALMONN phone features')
                                    | legend_display_dict(morpheme_name, 'SALMONN morpheme features')
                                    | legend_display_dict(wordpiece_name, 'SALMONN wordpiece features'))
        
        fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/tvl.png")

    elif transform_family_type == 'all_level_sensor':

        expression_data_tvl = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/en_all/all_tvl_gridsearch.nkg')
        tvl_name = expression_data_tvl.transforms
        IL_name = [i for i in tvl_name if i != 'STL']
        STL_name = ['STL']

        # import ipdb; ipdb.set_trace()

        expression_data_morpheme = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_morpheme/expression_set')
        morpheme_name = expression_data_morpheme.transforms
        for i in range(len(morpheme_name)):
            expression_data_morpheme.rename({morpheme_name[i]:f'{morpheme_name[i]}_morpheme'})
        expression_data_wordpiece = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_wordpiece/expression_set')
        wordpiece_name = expression_data_wordpiece.transforms
        for i in range(len(wordpiece_name)):
            expression_data_wordpiece.rename({wordpiece_name[i]:f'{wordpiece_name[i]}_wordpiece'})
        expression_data_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_phone/expression_set')
        phone_name = expression_data_phone.transforms
        for i in range(len(phone_name)):
            expression_data_phone.rename({phone_name[i]:f'{phone_name[i]}_phone'})
        expression_data_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_word/expression_set')
        word_name = expression_data_word.transforms
        for i in range(len(word_name)):
            expression_data_word.rename({word_name[i]:f'{word_name[i]}_word'})

        expression_data = expression_data_tvl + expression_data_word + expression_data_morpheme + expression_data_wordpiece + expression_data_phone

        save_expression_set(expression_data,'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/all_sensor_transforms.nkg')


        # fig = expression_plot(expression_data, paired_axes=False, show_legend=True,
        #                         color=constant_color_dict(word_name, color= 'red')
        #                             | constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink')
        #                             | constant_color_dict(phone_name, color='green')
        #                             | constant_color_dict(morpheme_name, color='blue')
        #                             | constant_color_dict(wordpiece_name, color='orange'),
        #                         legend_display=legend_display_dict(word_name, 'SALMONN word features')
        #                             | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness transform')
        #                             | legend_display_dict(phone_name, 'SALMONN phone features')
        #                             | legend_display_dict(morpheme_name, 'SALMONN morpheme features')
        #                             | legend_display_dict(wordpiece_name, 'SALMONN wordpiece features'))
        
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/further_results/all_sensor.png")


if __name__ == '__main__':
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    main()
