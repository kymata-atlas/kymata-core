from logging import basicConfig, INFO
from pathlib import Path
from os import path

from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set
from kymata.plot.plot import expression_plot, legend_display_dict
from kymata.plot.color import constant_color_dict, gradient_color_dict

import time
from tqdm import tqdm

def main():

    function_family_type = 'simple' # 'standard' or 'ANN' or 'simple'
    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-core", "kymata-core-data", "output")
    # path_to_nkg_files = '/imaging/woolgar/projects/Tianyi/kymata-core/kymata-core-data/output'

    # template invoker for printing out expression set .nkgs

    if function_family_type == 'simple':

        # expression_data  = load_expression_set(Path(path_to_nkg_files, 'model.decoder.layers.31.nkg'))

        # expression_data  = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/phonetics/brain/expression_set/phone_56_gridsearch.nkg')

        # phonetic_func = expression_data.functions

        expression_data = load_expression_set('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/en_all/all_tvl_gridsearch.nkg')

        # for func in expression_data.functions:
        #     fig = expression_plot(expression_data, show_only=func, paired_axes=False, minimap=False, show_legend=True,)
        #     fig.savefig(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/phonetics/brain/plot/{func}.png")

        # fig = expression_plot(expression_data, show_only=expression_data.functions[18:], paired_axes=False, minimap=False, show_legend=True,)
                            #   color=gradient_color_dict(expression_data.functions[:18], start_color = 'blue', stop_color="red"))

        fig = expression_plot(expression_data, paired_axes=False, minimap=False, show_legend=True, 
                              color=gradient_color_dict(['IL1', 'IL2', 'IL3', 'IL4', 'IL5','IL6', 'IL7', 'IL8', 'IL9'], start_color = 'blue', stop_color="purple")
                              | constant_color_dict(['IL'], 'red')
                              | constant_color_dict(['STL'], 'pink'))
                            #   | constant_color_dict(phonetic_func, 'green'))

        fig.savefig("/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/en_all/all_tvl_recolour.png")

    elif function_family_type == 'standard':

        expression_data  = load_expression_set(Path(path_to_nkg_files, 'russian_incremental/first_14_rus_gridsearch.nkg'))

        # import ipdb;ipdb.set_trace()

        fig = expression_plot(expression_data, paired_axes=True, minimap=False, show_legend=True, 
                              color=gradient_color_dict(['IL1', 'IL2', 'IL3', 'IL4', 'IL5','IL6', 'IL7', 'IL8', 'IL9'], start_color = 'blue', stop_color="purple")
                              | constant_color_dict(['IL'], 'red')
                              | constant_color_dict(['STL'], 'pink'))

        fig.savefig("expression_plot.png")

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
