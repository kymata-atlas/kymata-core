#!/usr/bin/python

import sys
import os
import pdb
import pylab as pl
import mne
import numpy as np
from mne.fiff import Evoked
from mne.minimum_norm import apply_inverse, read_inverse_operator


participants = [
                    'meg10_0003',
                    'meg10_0006',
                    'meg10_0007',
                    'meg10_0009',
                    'meg10_0011',
                    'meg10_0013',
                    'meg10_0019',
                    'meg10_0020',
                    'meg10_0021',
                    'meg10_0022',
                    'meg10_0028',
                    'meg10_0039',
                    'meg10_0040',
                    'meg10_0041',
                    'meg10_0043',
                    'meg10_0045',
                    'meg10_0061',
                    'meg10_0063',
                    'meg10_0073',
                    'meg10_0075'      
]


f = open('/imaging/at03/NKG_Data_Sets/VerbphraseMEG/scripts/Stimuli-Verbphrase-MEG-Single-col.txt', 'r')
words = list(f.read().split())

data_path = '/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG/'
subjects_dir = '/imaging/at03/NKG_Data_Sets/VerbphraseMEG/nme_subject_dir'

snr = 1
lambda2 = 1.0 / snr ** 2

for p in participants:

	# Make temporary STC to get it's vertices

	inverse_operator = read_inverse_operator((data_path + '2-sensor-data/inverse-operators/' + p + '_3L-corFM-loose02-nodepth-reg-inv.fif'))
	if os.path.exists((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[0] + '.fif')) == True:
		evoked = Evoked((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[0] + '.fif'))
	else:
		evoked = Evoked((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[3] + '.fif'))
	stc_from = apply_inverse(evoked, inverse_operator, lambda2, "MNE", pick_normal=False) # don't matter what this is

	foward = mne.read_forward_solution(data_path + '2-sensor-data/forward_models/' + p + '_5-3L-fwd.fif')
	noise_cov = mne.read_cov(data_path + '2-sensor-data/averaged-noise-covarience-files/' + p + '_gcov.fif')

	# First compute morph matices for participant	
	subject_to = 'average'
	subject_from = (p.split('_'))[1]
	vertices_to = mne.grade_to_vertices(subject_to, grade=4, subjects_dir=subjects_dir)
	morph_mat = mne.compute_morph_matrix(subject_from, subject_to, stc_from.vertno, vertices_to, subjects_dir=subjects_dir)

	# Compute source stcs
	for w in words:
		if os.path.exists((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + w + '.fif')) == True:
			# Apply Inverse
			evoked = Evoked((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + w + '.fif'), baseline=(-0.2, 0))
			noise_cov = mne.cov.regularize(noise_cov, evoked.info, mag=0.05, grad=0.05, eeg=0.1)
			data_cov = mne.computer_covariance(x, tmin = 0.04, tmax=0.15)			
			stc_from = lcmv(evoked, foward, noise_cov, data_cov, reg=0.01)
			# Morph to average
			stc_morphed = mne.morph_data_precomputed(subject_from, subject_to, stc_from, vertices_to, morph_mat)
			stc_morphed.save((data_path + '/3-single-trial-source-data/vert2562-smooth5-LCMV-signed/' + p + '-' + w))
	
