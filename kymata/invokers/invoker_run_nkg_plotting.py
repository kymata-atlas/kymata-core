from logging import basicConfig, INFO
from pathlib import Path
import os
from os import path
import numpy as np

from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set, save_expression_set
from kymata.plot.plot import expression_plot, legend_display_dict
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

    transform_family_type = 'simple' # 'standard' or 'ANN' or 'simple'
    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-core", "kymata-core-data", "output")
    # path_to_nkg_files = '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output'

    # template invoker for printing out expression set .nkgs

    if transform_family_type == 'simple':

        colours = {'Lexical Features': (0.4, 0, 0.6), 
            'Semantic Features': (0.6, 0, 0.4), 
            'Part of Speech': (0.8, 0, 0.2), 
            'Syntactic Features': (1, 0, 0),
            'Articulatory Features': (0, 0.6, 0.4), 
            'Phoneme Identities': (0, 1, 0)}

        # morpheme_manual = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/feats/morpheme/sensor/is_root_0_gridsearch.nkg')
        # morpheme_manual_name = morpheme_manual.transforms
        # morpheme_salmonn = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_7b_morpheme/expression_set')
        # morpheme_salmonn_name = morpheme_salmonn.transforms

        # fig = expression_plot(morpheme_manual + morpheme_salmonn, paired_axes=True, minimap=True, show_legend=True,
        #                       legend_display=legend_display_dict(morpheme_salmonn_name, 'Salmonn morpheme features')
        #                         | legend_display_dict(morpheme_manual_name, 'Manual morpheme features'))
        
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_vs_manual_morpheme_sensor.png")

        # expression_data_salmonn_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/word_source')
        expression_data_salmonn_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron')
        # expression_data_salmonn_word += load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/word_source')
        # expression_data_salmonn_word = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/feats/source/syntax/expression_set/syntax_4_gridsearch.nkg')
        # expression_data_salmonn_word += load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/feats/source/word/expression_set/word_22_gridsearch.nkg')
        # expression_data_salmonn_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron_whisper_v2')
        salmonn_word_name = expression_data_salmonn_word.transforms
        # expression_data_salmonn_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/phone_source')
        expression_data_salmonn_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron_phone')
        # expression_data_salmonn_phone += load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/phone_source')
        # expression_data_salmonn_phone = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/feats/source/phonetics/expression_set/phone_56_gridsearch.nkg')
        salmonn_phone_name = expression_data_salmonn_phone.transforms
        # expression_data_salmonn_morpheme = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/morpheme_source')
        # salmonn_morpheme_name = expression_data_salmonn_morpheme.transforms

        # import ipdb;ipdb.set_trace()

        expression_data_word_manual = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/feats/source/word/expression_set/word_22_gridsearch.nkg')
        expression_data_word_manual += load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/feats/source/syntax/expression_set/syntax_4_gridsearch.nkg')
        expression_data_phone_manual = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/feats/source/phonetics/expression_set/phone_56_gridsearch.nkg')
        # expression_data_morpheme_manual = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/feats/morpheme')
        manual_word_name = expression_data_word_manual.transforms
        manual_phone_name = expression_data_phone_manual.transforms
        # manual_morpheme_name = expression_data_morpheme_manual.transforms

        # import ipdb;ipdb.set_trace()

        expression_data_tvl = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/english_TVL_family_source_baseline_derangments_6.nkg')
        tvl_name = expression_data_tvl.transforms

        IL_name = ['IL']
        STL_name = ['STL']
        IL_channel_name = [i for i in tvl_name if i != 'STL' and i != 'IL']

        expression_data_whisper_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron_whisper_v2')
        whisper_word_name = expression_data_whisper_word.transforms

        # art_npy = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/art_sig.npy')
        # art_name = [f'layer{i[0]}_{i[1]}' for i in art_npy.tolist()]
        # id_npy = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/id_sig.npy')
        # id_name = [f'layer{i[0]}_{i[1]}' for i in id_npy.tolist()]
        # lex_npy = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/lex_sig.npy')
        # lex_name = [f'layer{i[0]}_{i[1]}' for i in lex_npy.tolist()]
        # pos_npy = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/pos_sig.npy')
        # pos_name = [f'layer{i[0]}_{i[1]}' for i in pos_npy.tolist()]
        # sem_npy = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/sem_sig.npy')
        # sem_name = [f'layer{i[0]}_{i[1]}' for i in sem_npy.tolist()]
        # syn_npy = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/syn_sig.npy')
        # syn_name = [f'layer{i[0]}_{i[1]}' for i in syn_npy.tolist()]

        # save_expression_set(expression_data_tvl +  expression_data_salmonn_word + expression_data_salmonn_phone
        #                       + expression_data_word_manual + expression_data_phone_manual, '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_feats.nkg')

        # fig = expression_plot(expression_data_tvl +  expression_data_salmonn_word + expression_data_salmonn_phone
        #                       + expression_data_word_manual + expression_data_phone_manual,
        #                       paired_axes=True, minimap_view='ventral' , minimap='large', show_legend=False, show_only=syn_name, ylim=-125,
        #                     color=constant_color_dict(syn_name, color= colours['Syntactic Features']),)

        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/all_feats_syn_ventral.png")

        # fig = expression_plot(expression_data_salmonn_morpheme, paired_axes=True, minimap=True, show_legend=True)
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/morpheme_source/salmonn_morpheme.png")

        fig = expression_plot(expression_data_tvl + expression_data_word_manual + expression_data_phone_manual,
                # expression_data_salmonn_word + expression_data_salmonn_phone + expression_data_whisper_word,
                paired_axes=True, minimap='large', show_legend=True, show_only=manual_phone_name+manual_word_name, ylim=-125,
                color=constant_color_dict(manual_word_name, color='red')
                    | gradient_color_dict(['IL1', 'IL2', 'IL3', 'IL4', 'IL5','IL6', 'IL7', 'IL8', 'IL9'], start_color = 'purple', stop_color="blue")
                    | constant_color_dict(IL_name, color='#4320aa')
                    | constant_color_dict(STL_name, color='#ca8bb5')
                    | constant_color_dict(manual_phone_name, color='green'),)
                # legend_display=legend_display_dict(salmonn_word_name, 'Salmonn word features')
                #     | legend_display_dict(IL_channel_name, 'Tonotopic Instantaneous Loudness transforms')
                #     | legend_display_dict(IL_name, 'Instantaneous Loudness transform')
                #     | legend_display_dict(STL_name, 'Short-Term Loudness transform')
                #     | legend_display_dict(salmonn_phone_name, 'Salmonn phone features'))
        fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/figure_2_man_only.png")

        # fig = expression_plot(expression_data_salmonn_word + expression_data_tvl + expression_data_salmonn_phone + expression_data_word_manual + expression_data_phone_manual, paired_axes=True, minimap=True, show_legend=True, show_only=manual_word_name+manual_phone_name, ylim=-100,
        # # fig = expression_plot(expression_data_salmonn_word + expression_data_tvl, paired_axes=True, minimap=True, show_legend=True,
        #                         color=constant_color_dict(manual_word_name, color= 'red')
        #                             | constant_color_dict(IL_channel_name, color= '#941de0')
        #                             | constant_color_dict(IL_name, color= '#4320aa')
        #                             | constant_color_dict(STL_name, color= '#ca8bb5')
        #                             | constant_color_dict(manual_phone_name, color='green'),
        #                         legend_display=legend_display_dict(manual_word_name, 'Manual word features')
        #                             | legend_display_dict(IL_channel_name, 'Tonotopic Instantaneous Loudness transforms')
        #                             | legend_display_dict(IL_name, 'Instantaneous Loudness transform')
        #                             | legend_display_dict(STL_name, 'Short-Term Loudness transform')
        #                             | legend_display_dict(manual_phone_name, 'Manual phone features'))
        #                         # color=constant_color_dict(lex_name, color= colours['Lexical Features'])
        #                         #     | constant_color_dict(sem_name, color= colours['Semantic Features'])
        #                         #     | constant_color_dict(pos_name, color= colours['Part of Speech'])
        #                         #     | constant_color_dict(syn_name, color= colours['Syntactic Features']),
        #                         # legend_display=legend_display_dict(lex_name, 'Lexical Features')
        #                         #     | legend_display_dict(sem_name, 'Semantic Features')
        #                         #     | legend_display_dict(pos_name, 'Part of Speech')
        #                         #     | legend_display_dict(syn_name, 'Syntactic Features'))
        #                         # color=constant_color_dict(art_name, color= colours['Articulatory Features'])
        #                         #     | constant_color_dict(id_name, color= colours['Phoneme Identities']),
        #                         # legend_display=legend_display_dict(art_name, 'Articulatory Features')
        #                         #     | legend_display_dict(id_name, 'Phoneme Identities'))
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/salmonn_vs_manual_part_source_show_manual.png")


    elif transform_family_type == 'standard':

        # expression_data  = load_expression_set(Path(path_to_nkg_files, 'russian_incremental/first_14_rus_gridsearch.nkg'))

        # # import ipdb;ipdb.set_trace()

        # fig = expression_plot(expression_data, paired_axes=True, minimap=False, show_legend=True, 
        #                       color=gradient_color_dict(['IL1', 'IL2', 'IL3', 'IL4', 'IL5','IL6', 'IL7', 'IL8', 'IL9'], start_color = 'blue', stop_color="purple")
        #                       | constant_color_dict(['IL'], 'red')
        #                       | constant_color_dict(['STL'], 'pink'))

        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_paper/single_neuron_word_source.png")

        # expression_data_salmonn_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_first_speech_paper/word_source')
        expression_data_salmonn_word_13b = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_first_speech_paper/single_neuron_13b')
        word_name_13b = expression_data_salmonn_word_13b.transforms
        expression_data_salmonn_word_7b = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_first_speech_paper/single_neuron')
        word_name_7b = expression_data_salmonn_word_7b.transforms
        # expression_data_salmonn_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_first_speech_paper/phone_source')
        expression_data_salmonn_phone_13b = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_first_speech_paper/single_neuron_13b_phone')
        phone_name_13b = expression_data_salmonn_phone_13b.transforms
        expression_data_salmonn_phone_7b = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_first_speech_paper/single_neuron_phone')
        phone_name_7b = expression_data_salmonn_phone_7b.transforms
        expression_data_tvl = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/english_TVL_family_source_baseline_derangments_6.nkg')
        tvl_name = expression_data_tvl.transforms
        IL_name = [i for i in tvl_name if i != 'STL']
        STL_name = ['STL']
        # import ipdb;ipdb.set_trace()
        fig = expression_plot(expression_data_salmonn_word_13b + expression_data_tvl + expression_data_salmonn_phone_13b + expression_data_salmonn_word_7b + expression_data_salmonn_phone_7b, paired_axes=True, minimap='large', show_legend=False, show_only=word_name_7b + phone_name_7b,
                                color=constant_color_dict(word_name_7b, color= 'red')
                                    # | constant_color_dict(tvl_name, color= 'yellow')
                                    | constant_color_dict(IL_name, color= 'purple')
                                    | constant_color_dict(STL_name, color= 'pink')
                                    | constant_color_dict(phone_name_7b, color='green'),
                                legend_display=legend_display_dict(word_name_13b, 'SALMONN word features')
                                    # | legend_display_dict(tvl_name, 'TVL transforms')
                                    | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
                                    | legend_display_dict(STL_name, 'Short Term Loudness transform')
                                    | legend_display_dict(phone_name_13b, 'SALMONN phone features'))
        fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/first_speech_first_speech_paper/salmonn_7b_vs_13b_show_7b.png")

    elif transform_family_type == 'ANN':

        start = time.time()

        path_to_nkg_files = Path(path_to_nkg_files, 'whisper_large_multi')

        # transform to find all files ending with _gridsearch.nkg
        def find_nkg_files(directory):
            return list(directory.glob('**/*_gridsearch.nkg'))

        # Find all matching files in the specified directories
        nkg_files = find_nkg_files(path_to_nkg_files)

        # Initialize expression_data
        expression_data = None

        # Loop through the file paths and load the data
        for file_path in tqdm(nkg_files):
            data = load_expression_set(file_path)
            if expression_data is None:
                expression_data = data
            else:
                expression_data += data

        # conv1_list = []
        # conv2_list = []
        # encoder0_list = []
        # encoder1_list = []
        # encoder2_list = []
        # encoder3_list = []
        # encoder4_list = []
        # encoder5_list = []

        # # Loop through the range from 0 to 511
        # for i in range(512):
        #     conv1 = f"model.encoder.conv1_{i}"
        #     conv1_list.append(conv1)

        # for i in range(512):
        #     conv2 = f"model.encoder.conv2_{i}"
        #     conv2_list.append(conv2)

        # for i in range(512):
        #     encoder0 = f"model.encoder.layers.0.final_layer_norm_{i}"
        #     encoder0_list.append(encoder0)

        # for i in range(512):
        #     encoder1 = f"model.encoder.layers.1.final_layer_norm_{i}"
        #     encoder1_list.append(encoder1)

        # for i in range(512):
        #     encoder2 = f"model.encoder.layers.2.final_layer_norm_{i}"
        #     encoder2_list.append(encoder2)

        # for i in range(512):
        #     encoder3 = f"model.encoder.layers.3.final_layer_norm_{i}"
        #     encoder3_list.append(encoder3)

        # for i in range(512):
        #     encoder4 = f"model.encoder.layers.4.final_layer_norm_{i}"
        #     encoder4_list.append(encoder4)

        # for i in range(512):
        #     encoder5 = f"model.encoder.layers.5.final_layer_norm_{i}"
        #     encoder5_list.append(encoder5)

        expression_plot(expression_data,
                        ylim=-400,
                        xlims=(-200, 800),
                        save_to=Path(Path(path.abspath("")).parent, "kymata-core/kymata-core-data", "output/whisper_large_all_expression.jpg"),
                        show_legend=False,)
                        # color= constant_color_dict(conv1_list, color='red')
                        #         | constant_color_dict(conv2_list, color='green')
                        #         | constant_color_dict(encoder0_list, color='blue')
                        #         | constant_color_dict(encoder1_list, color='cyan')
                        #         | constant_color_dict(encoder2_list, color='magenta')
                        #         | constant_color_dict(encoder3_list, color='yellow')
                        #         | constant_color_dict(encoder4_list, color='orange')
                        #         | constant_color_dict(encoder5_list, color='purple'),
                        # legend_display=legend_display_dict(conv1_list, 'Conv layer 1')
                        #                | legend_display_dict(conv1_list, 'Conv layer 1')
                        #                | legend_display_dict(conv2_list, 'Conv layer 2')
                        #                | legend_display_dict(encoder0_list, 'Encoder layer 1')
                        #                | legend_display_dict(encoder1_list, 'Encoder layer 2')
                        #                | legend_display_dict(encoder2_list, 'Encoder layer 3')
                        #                | legend_display_dict(encoder3_list, 'Encoder layer 4')
                        #                | legend_display_dict(encoder4_list, 'Encoder layer 5')
                        #                | legend_display_dict(encoder5_list, 'Encoder layer 6'))

        total_time_in_seconds = time.time() - start
        print(f'Time taken for code to run: {time.strftime("%H:%M:%S", time.gmtime(total_time_in_seconds))} ({total_time_in_seconds:.4f}s)')

if __name__ == '__main__':
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    main()
