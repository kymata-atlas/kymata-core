from pathlib import Path
from os import path

from kymata.io.config import load_config, get_root_dir
from kymata.io.nkg import load_expression_set
from kymata.plot.plot import expression_plot

# template invoker for printing out expression set .nkgs
def main():
    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset4.yaml")))

    data_root_dir = get_root_dir(config)

    mri_structurals_directory = config['mri_structurals_directory']

    path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-toolbox-data", "output")

    expression_data = load_expression_set(Path( path_to_nkg_files, "IL_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "STL_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "IL1_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "IL2_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "IL3_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "IL4_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "IL5_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "IL6_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "IL7_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "IL8_gridsearch.nkg"))
    expression_data += load_expression_set(Path( path_to_nkg_files, "IL9_gridsearch.nkg"))

    expression_plot(expression_data,
                    color = {
                        'IL': '#b11e34',
                        'IL1': '#a201e9',
                        'IL2': '#a201e9',
                        'IL3': '#a201e9',
                        'IL4': '#a201e9',
                        'IL5': '#a201e9',
                        'IL6': '#a201e9',
                        'IL7': '#a201e9',
                        'IL8': '#a201e9',
                        'IL9': '#a201e9',
                        'STL': '#d388b5'
                        },
                    minimap = {
                        'data_root_dir': data_root_dir,
                        'mri_structurals_directory': mri_structurals_directory
                        }
                    )