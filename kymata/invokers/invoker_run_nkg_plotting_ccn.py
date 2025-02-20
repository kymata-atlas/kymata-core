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

    transform_family_type = 'standard' # 'standard' or 'ANN' or 'simple'
    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-core", "kymata-core-data", "output")
    # path_to_nkg_files = '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output'

    # template invoker for printing out expression set .nkgs

    if transform_family_type == 'simple':

        pass


    elif transform_family_type == 'standard':

        dataset = 'ru_en'

        expression_data_tvl  = load_expression_set(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ccn_paper/{dataset}/all/tvl/all_tvl_gridsearch.nkg')

        # import ipdb;ipdb.set_trace()

        dec_neurons = np.load(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ccn_paper/{dataset}/all/dec_neurons.npy')

        expression_data_whisper = None

        for i in tqdm(range(dec_neurons.shape[0])):
            expression_data_whisper_all =  load_expression_set(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ccn_paper/{dataset}/all/expression_set/model.decoder.layers.{int(dec_neurons[i, 0]-32)}.fc2/model.decoder.layers.{int(dec_neurons[i, 0]-32)}.fc2_1279_gridsearch.nkg')
            if expression_data_whisper is None:
                expression_data_whisper = expression_data_whisper_all[f'model.decoder.layers.{int(dec_neurons[i, 0]-32)}.fc2_{int(dec_neurons[i, 1])}']
            else:
                expression_data_whisper += expression_data_whisper_all[f'model.decoder.layers.{int(dec_neurons[i, 0]-32)}.fc2_{int(dec_neurons[i, 1])}']

        expression_data_whisper_enc = None

        enc_neurons = np.load(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ccn_paper/{dataset}/all/enc_neurons.npy')

        for i in tqdm(range(enc_neurons.shape[0])):
            expression_data_whisper_all_enc =  load_expression_set(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ccn_paper/{dataset}/all/expression_set/model.encoder.layers.{int(enc_neurons[i, 0])}.fc2/model.encoder.layers.{int(enc_neurons[i, 0])}.fc2_1279_gridsearch.nkg')
            if expression_data_whisper_enc is None:
                expression_data_whisper_enc = expression_data_whisper_all_enc[f'model.encoder.layers.{int(enc_neurons[i, 0])}.fc2_{int(enc_neurons[i, 1])}']
            else:
                expression_data_whisper_enc += expression_data_whisper_all_enc[f'model.encoder.layers.{int(enc_neurons[i, 0])}.fc2_{int(enc_neurons[i, 1])}']

        tvl_name = expression_data_tvl.transforms
        whisper_name = expression_data_whisper.transforms
        whisper_enc_name = expression_data_whisper_enc.transforms

        fig = expression_plot(expression_data_tvl + expression_data_whisper_enc + expression_data_whisper, paired_axes=True, minimap=False, show_legend=True,
                      color=constant_color_dict(tvl_name, (0.5, 0.5, 0.5, 0.1))  # RGBA for transparent grey
                      | constant_color_dict(whisper_name, 'green')
                      | constant_color_dict(whisper_enc_name, 'blue'),
                      legend_display=legend_display_dict(tvl_name, 'Loudness transforms')
                           | legend_display_dict(whisper_name, 'Whisper decoder activations')
                           | legend_display_dict(whisper_enc_name, 'Whisper encoder activations'))

        fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ccn_paper/{dataset}/all/exp_incl_enc.png")

    elif transform_family_type == 'ANN':

        start = time.time()

        path_to_nkg_files = Path('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/ru_en/all_pilots/expression_set')

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

        decoder_list = []
        encoder_list = []

        for i in range(32):
            for j in range(1280):
                decoder = f"model.decoder.layers.{i}.fc2_{j}_gridsearch"
                decoder_list.append(decoder)

        for i in range(32):
            for j in range(1280):
                encoder = f"model.encoder.layers.{i}.fc2_{j}_gridsearch"
                encoder_list.append(encoder)

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
                        # ylim=-400,
                        xlims=(-200, 800),
                        save_to='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/ru_en/all_pilots/ru_en_all_pilots.png',
                        show_legend=False,
                        color=constant_color_dict(decoder_list, color='red')
                            | constant_color_dict(encoder_list, color='green'),
                        legend_display=legend_display_dict(decoder_list, 'Decoder')
                            | legend_display_dict(encoder_list, 'Encoder'))
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
