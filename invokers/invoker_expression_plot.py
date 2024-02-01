from os import path
from pathlib import Path
from tempfile import NamedTemporaryFile

from kymata.datasets.sample import KymataMirror2023Q3Dataset, TVLInsLoudnessOnlyDataset, TVLDeltaInsTC1LoudnessOnlyDataset
from kymata.io.nkg import save_expression_set, load_expression_set
from kymata.entities.expression import HexelExpressionSet, SensorExpressionSet
from kymata.plot.plot import expression_plot


output_path = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output'
expression_data_new_results = load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL1_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL2_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL3_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL4_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL5_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL6_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL7_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL8_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/IL9_gridsearch.nkg')
expression_data_new_results += load_expression_set(from_path_or_file='/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output/STL_gridsearch.nkg')

# expression_plot(expression_data_new_results)
save_expression_set(expression_data_new_results, '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-toolbox/kymata-toolbox-data/output_expression')