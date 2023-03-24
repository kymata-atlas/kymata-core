#!/usr/bin/python

import sys
sys.path.append('/imaging/local/software/mne_python/v0.9full')
sys.path.append('/imaging/local/software/python_packages/scikit-learn/v0.14.1')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.4')
import os
import pdb
import pylab as pl
import mne
import numpy as np
from mne import read_evokeds
from mne.minimum_norm import apply_inverse, read_inverse_operator


participants = [
                    'meg15_0045',
                    'meg15_0051',
                    'meg15_0054',
                    'meg15_0055',
                    'meg15_0056',
                    'meg15_0058',
                    'meg15_0060',
                    'meg15_0065',
                    'meg15_0066',
                    'meg15_0070',
                    'meg15_0071',
                    'meg15_0072',
                    'meg15_0079',
                    'meg15_0081',
                    'meg15_0082', 
]
inputstream = 'visual'

f = open('/imaging/at03/NKG_Data_Sets/DATASET_3-01_visual-and-auditory/items.txt', 'r')
words = list(f.read().split())

stcdir = '/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/4-single-trial-source-data/vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone/' + inputstream + '/'


for w in words:
	fname = os.path.join( stcdir, '%s-' + w + '-lh.stc' )
	stcs = [mne.read_source_estimate(fname % subject,subject='fsaverage') for subject in participants]

	#take mean average
	stc_avg = reduce(lambda x, y:x+y,stcs)
	stc_avg /= len(stcs)
	stc_avg.save(('/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/5-averaged-by-trial-data/vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone/' + inputstream + '/' + w))

	
