from pandas import DataFrame
from kymata.io.nkg import load_expression_set
from kymata.entities.expression import SensorExpressionSet
from tqdm import tqdm

path_to_nkg_files = '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/individual/encoder/expression_set/participant_01/model.encoder.layers.10.final_layer_norm_1279_gridsearch.nkg'
expression_data  = load_expression_set(path_to_nkg_files)
out_file = '/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/individual_log/slurm_log_encoder_2.txt'

with open(out_file, 'w') as file:
    for func in tqdm(expression_data.functions):
        data = expression_data.__getitem__(func).best_functions()
        min_row_index = data.iloc[:, -1].idxmin()
        min_row = data.loc[min_row_index]
        min_row_list = min_row.tolist()
        min_row_list[0], min_row_list[1], min_row_list[2] = min_row_list[1], min_row_list[2]*1000, min_row_list[0]
        min_row_list[-1] = -min_row_list[-1]  # Make the last item negative
        line = ', '.join(map(str, min_row_list))
        file.write(line + '\n')