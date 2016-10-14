print(__doc__)

## Removal mainline (50Hz) from EEG in MNE Python using notch filter

import numpy as np
import sys
sys.path.append('/imaging/local/software/mne_python/latest_v0.8')
sys.path.append('/imaging/local/software/python_packages/scikit-learn/v0.14.1')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.4')

import mne
from mne import io

participants = [
	'meg15_0082',
	'meg15_0086'
]


data_path = '/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/'

block = ['1', '2']

for b in block:
	for p in participants:

		raw_fname = data_path + '1-preprosessing/sss/' + p + '_nkg_part' + b + '_raw_sss_movecomp.fif'
		raw = io.Raw(raw_fname, preload=True)
		raw.notch_filter(50, picks=None, n_jobs=1)
		raw_ica_fname = data_path + '1-preprosessing/sss/' + p + '_nkg_part' + b + '_raw_sss_movecomp_EEGmainline.fif'
		raw.save(raw_ica_fname, overwrite=True)
	



