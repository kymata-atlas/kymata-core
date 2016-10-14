#!/usr/bin/python

import sys
import os
import pdb
import pylab as pl
import numpy as np
import mne
from mne import read_evokeds
from mne.minimum_norm import apply_inverse, read_inverse_operator


participants = [
                    'meg08_0320',
		     'meg08_0323',
		     'meg08_0324',
		     'meg08_0327',
		     'meg08_0348',
		     'meg08_0350',
		     'meg08_0363',
		     'meg08_0366',
		     'meg08_0371',
		     'meg08_0372',
		     'meg08_0377',
		     'meg08_0380',
		     'meg08_0397',
		     'meg08_0400',
		     'meg08_0401',
		     'meg08_0402'   
]


f = open('/imaging/at03/NKG_Data_Sets/LexproMEG/scripts/Stimuli-Lexpro-MEG-Single-col.txt', 'r')
words = list(f.read().split())

data_path = '/imaging/at03/NKG_Code_output/Version4_2/LexproMEG/'
subjects_dir = '/imaging/at03/NKG_Data_Sets/LexproMEG/nme_subject_dir'

#participants = [
#                    'meg10_0003',
#                    'meg10_0006',
#                    'meg10_0007',
#                    'meg10_0009',
#                    'meg10_0011',
#                    'meg10_0013',
#                    'meg10_0019',
#                    'meg10_0020',
#                    'meg10_0021',
#                    'meg10_0022',
#                    'meg10_0028',
#                    'meg10_0039',
#                    'meg10_0040',
#                    'meg10_0041',
#                    'meg10_0043',
#                    'meg10_0045',
#                    'meg10_0061',
#                    'meg10_0063',
#                    'meg10_0073',
#                    'meg10_0075'      
#]


#f = open('/imaging/at03/NKG_Data_Sets/VerbphraseMEG/scripts/Stimuli-Verbphrase-MEG-Single-col.txt', 'r')
#words = list(f.read().split())

#data_path = '/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG'
#subjects_dir = '/imaging/at03/NKG_Data_Sets/VerbphraseMEG/nme_subject_dir'

snr = 1
lambda2 = 1.0 / snr ** 2

for p in participants:

	# Make temporary STC to get it's vertices

	#inverse_operator = read_inverse_operator((data_path + '2-sensor-data/inverse-operators/' + p + '_ico-5-3L-corFM-loose02-nodepth-reg-inv-noMedWall-csd.fif'))
	inverse_operator = read_inverse_operator((data_path + '2-sensor-data/inverse-operators/' + p + '_3L-loose0.2-nodepth-reg-inv.fif'))
	if os.path.exists((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[0] + '.fif')) == True:
		evoked = read_evokeds((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[0] + '.fif'), condition=0, baseline=None)
	elif os.path.exists((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[1] + '.fif')) == True:
		evoked = read_evokeds((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[1] + '.fif'), condition=0, baseline=None)
	elif os.path.exists((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[2] + '.fif')) == True:
		evoked = read_evokeds((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[2] + '.fif'), condition=0, baseline=None)
	elif os.path.exists((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[3] + '.fif')) == True:
		evoked = read_evokeds((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[3] + '.fif'), condition=0, baseline=None)
	elif os.path.exists((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[4] + '.fif')) == True:
		evoked = read_evokeds((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[4] + '.fif'), condition=0, baseline=None)
	else:
		evoked = read_evokeds((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + words[5] + '.fif'), condition=0, baseline=None)
	stc_from = apply_inverse(evoked, inverse_operator, lambda2, "MNE", pick_ori=None) # don't matter what this is


	# First compute morph matices for participant	
	subject_to = 'average'
	#subject_to = 'fsaverage'
	subject_from = (p.split('_'))[1]
	vertices_to = mne.grade_to_vertices(subject_to, grade=5, subjects_dir=subjects_dir) #grade 4 is 2562
	morph_mat = mne.compute_morph_matrix(subject_from, subject_to, stc_from.vertno, vertices_to, subjects_dir=subjects_dir)

	# Compute source stcs
	for w in words:
		if os.path.exists((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + w + '.fif')) == True:
			# Apply Inverse
			evoked = read_evokeds((data_path + '2-sensor-data/fif-out-averaged/' + p + '-' + w + '.fif'), condition=0, baseline=(-0.2, 0))
			stc_from = apply_inverse(evoked, inverse_operator, lambda2, "MNE", pick_ori="normal")
			# Morph to average
			stc_morphed = mne.morph_data_precomputed(subject_from, subject_to, stc_from, vertices_to, morph_mat)
			stc_morphed.save((data_path + '/3-single-trial-source-data/vert10242-smooth5-nodepth-eliFM-snr1-signed/' + p + '-' + w))
			#stc_morphed.save((data_path + '/3-single-trial-source-data/vert10242-smooth5-nodepth-corFM-snr1-signed-fsaverage-noMedWall-csd-minus30/' + p + '-' + w))
	
