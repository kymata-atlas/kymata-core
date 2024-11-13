from logging import basicConfig, INFO
from pathlib import Path
import os
from os import path
import numpy as np

from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set
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

    function_family_type = 'simple' # 'standard' or 'ANN' or 'simple'
    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-core", "kymata-core-data", "output")
    # path_to_nkg_files = '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output'

    # template invoker for printing out expression set .nkgs

    if function_family_type == 'simple':

        # expression_data  = load_expression_set(Path(path_to_nkg_files, 'model.decoder.layers.31.nkg'))

        # expression_data  = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/phonetics/brain/expression_set/phone_56_gridsearch.nkg')

        # phonetic_func = expression_data.functions

        # expression_data = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/en_all/all_tvl_gridsearch.nkg')

        # base_folder = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron"
        # # base_folder = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron_sensor"

        # # Load all expression data from .nkg files
        # expression_data_salmonn = load_all_expression_data(base_folder)

        # # import ipdb;ipdb.set_trace()

        # salmonn_name = expression_data_salmonn.functions

        # # expression_data_word = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/feats/word/brain/expression_set/word_22_gridsearch.nkg')
        expression_data_word = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/feats/source/word/expression_set/word_22_gridsearch.nkg')
        word_name = expression_data_word.functions
        lex_name = word_name[:3]
        sem_name = word_name[3:6]
        pos_name = word_name[14:]

        # # expression_data_syntax = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/feats/syntax/brain/expression_set/syntax_4_gridsearch.nkg')
        expression_data_syntax = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/feats/source/syntax/expression_set/syntax_4_gridsearch.nkg')
        syntax_name = expression_data_syntax.functions

        # all_data = expression_data_salmonn + expression_data_word + expression_data_syntax
        # all_data = expression_data_word + expression_data_syntax

        # import ipdb;ipdb.set_trace()


        # for func in expression_data.functions:
        #     fig = expression_plot(expression_data, show_only=func, paired_axes=False, minimap=False, show_legend=True,)
        #     fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/phonetics/brain/plot/{func}.png")

        # fig = expression_plot(expression_data, show_only=expression_data.functions[18:], paired_axes=False, minimap=False, show_legend=True,)
                            #   color=gradient_color_dict(expression_data.functions[:18], start_color = 'blue', stop_color="red"))

        # fig = expression_plot(expression_data, paired_axes=True, minimap=False, show_legend=True,
                            #   | constant_color_dict(phonetic_func, 'green'))

        ### First plot
        expression_data  = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/feats/source/phonetics/expression_set/phone_56_gridsearch.nkg')
        phonetic_func = expression_data.functions
        expression_data_salmonn_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/phone_source')
        salmonn_name = expression_data_salmonn_phone.functions
        fig = expression_plot(expression_data + expression_data_salmonn_phone, paired_axes=True, minimap=False, show_legend=False,
                                color= constant_color_dict(phonetic_func, color='#98fb98')
                                    | constant_color_dict(salmonn_name, color= 'green'))
                                # legend_display=legend_display_dict(phonetic_func, 'Phoneme features')
                                #     | legend_display_dict(salmonn_name, 'SALMONN neurons'))
        # fig = expression_plot(all_data, paired_axes=True, minimap=True, show_legend=True)

        fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_all_phone_vs_feats_source_v3.png")

        expression_data_salmonn_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/word_source')
        salmonn_word_name = expression_data_salmonn_word.functions
        fig = expression_plot(expression_data_word + expression_data_syntax + expression_data_salmonn_word, paired_axes=True, minimap=False, show_legend=False, show_only=lex_name + sem_name + pos_name + syntax_name + salmonn_word_name,
                                color= constant_color_dict(lex_name + sem_name + pos_name + syntax_name, color='#ffcccb')
                                    | constant_color_dict(salmonn_word_name, color= 'red'))
                                # legend_display=legend_display_dict(lex_name + sem_name + pos_name + syntax_name, 'Word features')
                                #     | legend_display_dict(salmonn_word_name, 'SALMONN neurons'))
        # fig = expression_plot(all_data, paired_axes=True, minimap=True, show_legend=True)

        fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_all_vs_feats_source_v3.png")

        # expression_data_salmonn_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/word_source')
        # word_name = expression_data_salmonn_word.functions
        # expression_data_salmonn_phone = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/phone_source')
        # phone_name = expression_data_salmonn_phone.functions
        # expression_data_tvl = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/english_TVL_family_source_baseline.nkg')
        # tvl_name = expression_data_tvl.functions
        # IL_name = [i for i in tvl_name if i != 'STL']
        # STL_name = ['STL']
        # fig = expression_plot(expression_data_salmonn_word + expression_data_tvl + expression_data_salmonn_phone, paired_axes=True, minimap=False, show_legend=True,
        #                         color=constant_color_dict(word_name, color= 'red')
        #                             # | constant_color_dict(tvl_name, color= 'yellow')
        #                             | constant_color_dict(IL_name, color= 'purple')
        #                             | constant_color_dict(STL_name, color= 'pink')
        #                             | constant_color_dict(phone_name, color='green'),
        #                         legend_display=legend_display_dict(word_name, 'SALMONN word features')
        #                             # | legend_display_dict(tvl_name, 'TVL functions')
        #                             | legend_display_dict(IL_name, 'Instantaneous Loudness Functions')
        #                             | legend_display_dict(STL_name, 'Short Term Loudness Function')
        #                             | legend_display_dict(phone_name, 'SALMONN phone features'),
        #                         display_range=(40, 55))
        # fig = expression_plot(expression_data_tvl[40:55], paired_axes=True, minimap=False, show_legend=True)
        # fig = expression_plot(expression_data_tvl, paired_axes=True, minimap=False, show_legend=True, display_range=slice(40, 55))

        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_word_vs_phone_part_source.png")
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_word_vs_phone_vs_tvl_all_source_0_75.png")


        # ### Second plot
        # base_folder = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron_phone"
        # # base_folder = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron_sensor"
        # salmonn_id = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/id_sig.npy')
        # salmonn_art = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/art_sig.npy')
        # expression_data_salmonn_id = load_part_of_expression_data(base_folder, salmonn_id)
        # expression_data_salmonn_art = load_part_of_expression_data(base_folder, salmonn_art)
        # expression_data_salmonn_phone = expression_data_salmonn_id + expression_data_salmonn_art
        # id_name_salmonn = expression_data_salmonn_id.functions
        # art_name_salmonn = expression_data_salmonn_art.functions
        # fig = expression_plot(expression_data_salmonn_phone, paired_axes=True, minimap=True, show_legend=True,
        #                         color= constant_color_dict(id_name_salmonn, color='green')
        #                             | constant_color_dict(art_name_salmonn, color= 'red'),
        #                         legend_display=legend_display_dict(id_name_salmonn, 'Salmonn Phoneme Identities')
        #                             | legend_display_dict(art_name_salmonn, 'Salmonn Articulatory Features'))
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron_phone_source.png")


        # ### Third plot
        # expression_data_feats = expression_data_word + expression_data_syntax
        # fig = expression_plot(expression_data_feats, paired_axes=True, minimap=True, show_legend=True, show_only = lex_name + sem_name + pos_name + syntax_name,
        #                         color= constant_color_dict(syntax_name, color='green')
        #                             | constant_color_dict(lex_name, color= 'red')
        #                             | constant_color_dict(sem_name, 'blue')
        #                             | constant_color_dict(pos_name, 'pink'),
        #                         legend_display=legend_display_dict(syntax_name, 'Syntax')
        #                             | legend_display_dict(lex_name, 'Lexicon')
        #                             | legend_display_dict(sem_name, 'Semantics')
        #                             | legend_display_dict(pos_name, 'Part of Speech'))
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/feats_source.png")


        # ### Fourth plot
        # base_folder = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron"
        # salmonn_sem = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/sem_sig.npy')
        # salmonn_lex = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/lex_sig.npy')
        # salmonn_syn = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/syn_sig.npy')
        # salmonn_pos = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/neuron_picks/pos_sig.npy')
        # expression_data_salmonn_sem = load_part_of_expression_data(base_folder, salmonn_sem)
        # expression_data_salmonn_lex = load_part_of_expression_data(base_folder, salmonn_lex)
        # expression_data_salmonn_syn = load_part_of_expression_data(base_folder, salmonn_syn)
        # expression_data_salmonn_pos = load_part_of_expression_data(base_folder, salmonn_pos)
        # sem_name_salmonn = expression_data_salmonn_sem.functions
        # syn_name_salmonn = expression_data_salmonn_syn.functions
        # lex_name_salmonn = expression_data_salmonn_lex.functions
        # pos_name_salmonn = expression_data_salmonn_pos.functions
        # fig = expression_plot(expression_data_salmonn, paired_axes=True, minimap=True, show_legend=True,
        #                         color= constant_color_dict(syn_name_salmonn, color='green')
        #                             | constant_color_dict(lex_name_salmonn, color= 'red')
        #                             | constant_color_dict(sem_name_salmonn, 'blue')
        #                             | constant_color_dict(pos_name_salmonn, 'pink'),
        #                         legend_display=legend_display_dict(syn_name_salmonn, 'Salmonn Syntax')
        #                             | legend_display_dict(lex_name_salmonn, 'Salmonn Lexicon')
        #                             | legend_display_dict(sem_name_salmonn, 'Salmonn Semantics')
        #                             | legend_display_dict(pos_name_salmonn, 'Salmonn Part of Speech'))
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron_source.png")



        # ### Fifth plot
        # expression_data_phone_feats = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/feats/source/phonetics/expression_set/phone_56_gridsearch.nkg')
        # id_name = expression_data_phone_feats.functions[18:]
        # art_name = expression_data_phone_feats.functions[:18]
        # fig = expression_plot(expression_data_phone_feats, paired_axes=True, minimap=True, show_legend=True,
        #                         color= constant_color_dict(id_name, color='green')
        #                             | constant_color_dict(art_name, color= 'red'),
        #                         legend_display=legend_display_dict(id_name, 'Phoneme Identities')
        #                             | legend_display_dict(art_name, 'Articulatory Features'))
        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/feats_phone_source.png")        


        # ### Sixth plot
        # fig = expression_plot(expression_data_phone_feats + expression_data_salmonn_phone, paired_axes=True, minimap=True, show_legend=True,
        #                         color= constant_color_dict(expression_data_phone_feats.functions, color='green')
        #                             | constant_color_dict(expression_data_salmonn_phone.functions, color= 'red'),
        #                         legend_display=legend_display_dict(expression_data_phone_feats.functions, 'Phoneme features')
        #                             | legend_display_dict(expression_data_salmonn_phone.functions, 'SALMONN neurons'))
        # # fig = expression_plot(all_data, paired_axes=True, minimap=True, show_legend=True)

        # fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_vs_feats_phone_source.png")        





        # Load all expression data from .nkg files
        # expression_data_salmonn = load_all_expression_data(base_folder)

    elif function_family_type == 'standard':

        expression_data  = load_expression_set(Path(path_to_nkg_files, 'russian_incremental/first_14_rus_gridsearch.nkg'))

        # import ipdb;ipdb.set_trace()

        fig = expression_plot(expression_data, paired_axes=True, minimap=False, show_legend=True, 
                              color=gradient_color_dict(['IL1', 'IL2', 'IL3', 'IL4', 'IL5','IL6', 'IL7', 'IL8', 'IL9'], start_color = 'blue', stop_color="purple")
                              | constant_color_dict(['IL'], 'red')
                              | constant_color_dict(['STL'], 'pink'))

        fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/single_neuron_word_source.png")

    elif function_family_type == 'ANN':

        start = time.time()

        path_to_nkg_files = Path(path_to_nkg_files, 'whisper_large_multi')

        # Function to find all files ending with _gridsearch.nkg
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
