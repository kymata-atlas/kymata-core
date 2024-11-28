from os import path
from pathlib import Path
from tempfile import NamedTemporaryFile

from kymata.datasets.sample import KymataMirror2023Q3Dataset, TVLInsLoudnessOnlyDataset, TVLDeltaInsTC1LoudnessOnlyDataset
from kymata.io.nkg import save_expression_set, load_expression_set

## Download sample data. This cell can be ignored if you wish to load your own
## data from a gridsearch.

# set location of tutorial data
# sample_data_dir = Path(Path(path.abspath("")).parent, "kymata-core-data", "tutorial_nkg_data")
sample_data_dir = Path(Path(path.abspath("")).parent, "kymata-core-data")
sample_data_dir.mkdir(exist_ok=True)

# First we'll download a sample .nkg file which loads a range of functions,
# from the Kymata Research Group. nkg files contain both lefthand and
# righthand data for a set of functions

sample_dataset = KymataMirror2023Q3Dataset(data_root=sample_data_dir, download=True)
nkg_path = Path(sample_dataset.path, sample_dataset.filenames[0])
print(nkg_path.name)

# Second we will download two .nkg files which only contain one
# function each - 'ins_loudness' and 'd_ins_tc1_loudness':
ins_loudness_only_dataset = TVLInsLoudnessOnlyDataset(data_root=sample_data_dir, download=True)
ins_loudness_path = Path(ins_loudness_only_dataset.path, ins_loudness_only_dataset.filenames[0])

d_ins_tc1_loudness_only_dataset = TVLDeltaInsTC1LoudnessOnlyDataset(data_root=sample_data_dir, download=True)
d_ins_tc1_loudness_path = Path(d_ins_tc1_loudness_only_dataset.path, d_ins_tc1_loudness_only_dataset.filenames[0])

# Let's load the KymataMirror2023Q3 .nkg file. This contains around 30 functions.
expression_data_kymata_mirror = load_expression_set(from_path_or_file=nkg_path)

pass

# python invokers/run_gridsearch.py --config dataset4.yaml --input-stream auditory --function-path 'predicted_function_contours/GMSloudness/stimulisig' --function-name IL