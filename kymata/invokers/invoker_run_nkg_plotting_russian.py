from logging import basicConfig, INFO
from pathlib import Path
import os
from os import path
import numpy as np
from statistics import NormalDist

from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set
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

    transform_family_type = 'standard' # 'standard' or 'ANN' or 'simple'
    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-core", "kymata-core-data", "output")
    # path_to_nkg_files = '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output'

    # template invoker for printing out expression set .nkgs

    if transform_family_type == 'simple':

        pass


    elif transform_family_type == 'standard':

        expression_data_tvl  = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/en_ru_tvl_gridsearch.nkg')

        # import ipdb;ipdb.set_trace()

        dec_neurons = np.load('/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/decoder_text/dec_neurons.npy')

        expression_data_qwen = None

        for i in tqdm(range(dec_neurons.shape[0])):
            expression_data_qwen_all =  load_expression_set(f'/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/decoder_text/expression/layer{int(dec_neurons[i, 0])}/layer{int(dec_neurons[i, 0])}_3583_gridsearch.nkg')
            if expression_data_qwen is None:
                expression_data_qwen = expression_data_qwen_all[f'layer{int(dec_neurons[i, 0])}_{int(dec_neurons[i, 1])}']
            else:
                expression_data_qwen += expression_data_qwen_all[f'layer{int(dec_neurons[i, 0])}_{int(dec_neurons[i, 1])}']

        # expression_data_qwen = None

        # for i in tqdm(range(29)):
        #     expression_data_qwen_all =  load_expression_set(f'/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/decoder_text/expression/layer{i}/layer{i}_3583_gridsearch.nkg')
        #     if expression_data_qwen is None:
        #         expression_data_qwen = expression_data_qwen_all
        #     else:
        #         expression_data_qwen += expression_data_qwen_all

        # expression_data_qwen_enc = None

        # enc_neurons = np.load(f'/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/encoder/enc_neurons.npy')

        # for i in tqdm(range(enc_neurons.shape[0])):
        #     expression_data_qwen_all_enc =  load_expression_set(f'/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/encoder/expression/layer{int(enc_neurons[i, 0])}/layer{int(enc_neurons[i, 0])}_3583_gridsearch.nkg')
        #     if expression_data_qwen_enc is None:
        #         expression_data_qwen_enc = expression_data_qwen_all_enc[f'layer.{int(enc_neurons[i, 0])}_{int(enc_neurons[i, 1])}']
        #     else:
        #         expression_data_qwen_enc += expression_data_qwen_all_enc[f'layer.{int(enc_neurons[i, 0])}_{int(enc_neurons[i, 1])}']

        tvl_name = expression_data_tvl.transforms
        qwen_name = expression_data_qwen.transforms
        # qwen_enc_name = expression_data_qwen_enc.transforms

        alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
        thres = np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*370*3584*29)))))

        exp_all = expression_data_tvl + expression_data_qwen
        best_trans = exp_all.best_transforms()
        dec_trans = [i.transform for i in best_trans if i.logp_value <= thres and i.transform in qwen_name]

        import ipdb;ipdb.set_trace()

        fig = expression_plot(exp_all, paired_axes=True, minimap=None, show_legend=True, show_only=qwen_name,
                      color=constant_color_dict(tvl_name, (0.5, 0.5, 0.5, 0.1))  # RGBA for transparent grey
                      | constant_color_dict(qwen_name, 'red'),
                      legend_display=legend_display_dict(tvl_name, 'Loudness transforms')
                           | legend_display_dict(qwen_name, 'Qwen decoder activations'))

        fig.savefig('/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/decoder_text/exp_tvl_vs_dec_show_dec.png')

        fig = expression_plot(exp_all, paired_axes=True, minimap=None, show_legend=True,
                      color=constant_color_dict(tvl_name, (0.5, 0.5, 0.5, 0.1))  # RGBA for transparent grey
                      | constant_color_dict(qwen_name, 'red'),
                      legend_display=legend_display_dict(tvl_name, 'Loudness transforms')
                           | legend_display_dict(qwen_name, 'Qwen decoder activations'))

        fig.savefig('/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/decoder_text/exp_tvl_vs_dec.png')

        fig = expression_plot(expression_data_qwen, paired_axes=True, minimap=None, show_legend=True, show_only=qwen_name,
                      color=constant_color_dict(tvl_name, (0.5, 0.5, 0.5, 0.1))  # RGBA for transparent grey
                      | constant_color_dict(qwen_name, 'red'),
                      legend_display=legend_display_dict(tvl_name, 'Loudness transforms')
                           | legend_display_dict(qwen_name, 'Qwen decoder activations'))

        fig.savefig('/imaging/projects/cbu/kymata/analyses/tianyi/russian-english/kymata-core/kymata-core-data/output/qwen_english_russian/sensor/decoder_text/exp_dec.png')

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

        expression_plot(expression_data,
                        # ylim=-400,
                        xlims=(-200, 800),
                        save_to='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/ru_en/all_pilots/ru_en_all_pilots.png',
                        show_legend=False,
                        color=constant_color_dict(decoder_list, color='red')
                            | constant_color_dict(encoder_list, color='green'),
                        legend_display=legend_display_dict(decoder_list, 'Decoder')
                            | legend_display_dict(encoder_list, 'Encoder'))
        total_time_in_seconds = time.time() - start
        print(f'Time taken for code to run: {time.strftime("%H:%M:%S", time.gmtime(total_time_in_seconds))} ({total_time_in_seconds:.4f}s)')

if __name__ == '__main__':
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    main()
