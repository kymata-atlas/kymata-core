from logging import basicConfig, INFO
from pathlib import Path
import os
from os import path
import numpy as np

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

    transform_family_type = 'simple' # 'standard' or 'ANN' or 'simple' or 'all_level'
    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-core", "kymata-core-data", "output")
    # path_to_nkg_files = '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output'

    # template invoker for printing out expression set .nkgs

    if transform_family_type == 'simple':

        expression_data_ecog_word = load_all_expression_data('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ecog_language/word/expression')
        word_name = expression_data_ecog_word.transforms
        # phone_name = expression_data_salmonn_phone.transforms
        expression_data_tvl = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/ecog/kymata-core/kymata-core-data/output/sub-kmeans300/11_transforms_gridsearch.nkg')
        tvl_name = expression_data_tvl.transforms
        IL_name = [i for i in tvl_name if i != 'STL']
        STL_name = ['STL']
        fig = expression_plot(expression_data_ecog_word + expression_data_tvl, paired_axes=True, minimap=None, show_legend=True,
                                color=constant_color_dict(word_name, color= 'red')
                                    # | constant_color_dict(tvl_name, color= 'yellow')
                                    | constant_color_dict(IL_name, color= 'purple')
                                    | constant_color_dict(STL_name, color= 'pink'),
                                    # | constant_color_dict(phone_name, color='green'),
                                legend_display=legend_display_dict(word_name, 'ECOG word features')
                                    # | legend_display_dict(tvl_name, 'TVL transforms')
                                    | legend_display_dict(IL_name, 'Instantaneous Loudness transforms')
                                    | legend_display_dict(STL_name, 'Short Term Loudness transform'))
                                    # | legend_display_dict(phone_name, 'SALMONN phone features'))

        fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ecog_language/word/ecog_word_vs_tvl.png")


if __name__ == '__main__':
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    main()
