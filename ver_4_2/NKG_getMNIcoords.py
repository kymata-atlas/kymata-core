#!/usr/bin/python

import sys
sys.path.append('/imaging/local/software/mne_python/latest_v0.8')
sys.path.append('/imaging/local/software/python_packages/scikit-learn/v0.14.1')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.4')
import os
import pdb
import pylab as pl
import mne
import numpy as np
from mne.fiff import Evoked
from mne.minimum_norm import apply_inverse, read_inverse_operator

data_path = '/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG/'
subjects_dir = '/imaging/at03/NKG_Data_Sets/VerbphraseMEG/nme_subject_dir'

vertices = range(10)

for hemi in range(0,1):

	coordinates = mne.vertex_to_mni(vertices, hemi, 'average', subjects_dir, 'freesurfer')	

	print(coordinates)


