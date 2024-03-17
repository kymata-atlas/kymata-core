from pathlib import Path
from os import path
from kymata.io.nkg import load_expression_set
from kymata.plot.plot import expression_plot

# template invoker for printing out expression set .nkgs

path_to_nkg_files = Path(Path(path.abspath("")).parent, "kymata-toolbox/kymata-toolbox-data", "output/all_participant_sensor")

# expression_data = load_expression_set(Path( path_to_nkg_files, "d_IL_gridsearch.nkg"))
expression_data = load_expression_set(Path( path_to_nkg_files, "d_IL4_gridsearch.nkg"))
expression_data += load_expression_set(Path( path_to_nkg_files, "d_STL_gridsearch.nkg"))
# expression_data += load_expression_set(Path( path_to_nkg_files, "IL_gridsearch.nkg"))
expression_data += load_expression_set(Path( path_to_nkg_files, "IL4_gridsearch.nkg"))
expression_data += load_expression_set(Path( path_to_nkg_files, "STL_gridsearch.nkg"))

expression_plot(expression_data, color = {
                    # 'd_IL': '#1f77b4',
                    'd_IL4': '#ff7f0e',
                    'd_STL': '#2ca02c',
                    # 'IL': '#d62728',
                    'IL4': '#9467bd',
                    'STL': '#8c564b'
                  }, ylim=-200, save_to=Path(Path(path.abspath("")).parent, "kymata-toolbox/kymata-toolbox-data", "output/all_participant_erp.jpg"))