print(__doc__)

## ICA for blink removal in MNE Python
##
## Based on the script of Caroline Whiting, June 2014
## NOTE: need to run ipython --matplotlib=qt to begin


import sys
sys.path.append('/home/at03/Desktop/mne-python-master')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.4')
import numpy as np
import mne
from mne import read_cov
from mne.io import Raw
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs


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
	'meg15_0086' 
]


data_path = '/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/'

block = '1'

for p in participants:

	# Set up data
	raw_fname = data_path + '1-preprosessing/sss/' + p + '_nkg_part' + block + '_raw_sss_movecomp_tr.fif'
	raw = Raw(raw_fname, preload=True)
	raw.filter(1, 45, n_jobs=2)
	reject = dict(mag=4e-12, grad=4000e-13)
 
	picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')

	###############################################################################
	# 1) Fit ICA model 

	#check all options in https://github.com/mne-tools/mne-python/blob/master/mne/preprocessing/ica.py - lokts of them!!!!
 
	ica = ICA(n_components=0.95, method='extended-infomax')

	ica.fit(raw, picks=picks, decim=3,reject=reject)

	# maximum number of components to reject
	n_max_ecg, n_max_veog = 1,1


	###############################################################################
	# 2) Find ECG Artifacts in MEG

	# generate ECG epochs to improve detection by correlation
	ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks)

	ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')

	title = 'Sources related to %s artifacts (red)'
	ica.plot_scores(scores, exclude=ecg_inds, title=title % 'ecg')

	show_picks = np.abs(scores).argsort()[::-1][:5]

	ica.plot_sources(raw, show_picks, exclude=ecg_inds, title=title % 'ecg')
	ica.plot_components(ecg_inds, title=title % 'ecg', colorbar=True)

	ecg_inds = ecg_inds[:n_max_ecg]
	ica.exclude += ecg_inds


	# detect VEOG by correlation

#	eog_inds, scores = ica.find_bads_eog(raw)
#	ica.plot_scores(scores, exclude=eog_inds, title=title % 'eog')
#
#	show_picks = np.abs(scores).argsort()[::-1][:5]
#
#	ica.plot_sources(raw, show_picks, exclude=eog_inds, title=title % 'eog')
#	ica.plot_components(eog_inds, title=title % 'eog', colorbar=True)
#	
#	eog_inds = eog_inds[:n_max_veog]
#	ica.exclude += eog_inds

	###############################################################################
	# 3) Assess component selection and unmixing quality

	# estimate average artifact
	ecg_evoked = ecg_epochs.average()
	ica.plot_sources(ecg_evoked)  # plot ECG sources + selection
	#ica.plot_overlay(ecg_evoked)  # plot ECG cleaning


#	eog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5, picks=picks).average()
#	ica.plot_sources(eog_evoked, exclude=eog_inds)  # plot EOG sources + selection
#	ica.plot_overlay(eog_evoked, exclude=eog_inds)  # plot EOG cleaning

	# check the amplitudes do not change
	ica.plot_overlay(raw, start=40000,stop=45000)  # ECG artifacts remain

	raw2 = Raw(raw_fname, preload=True)
	ica.apply(raw2, copy=False)
	foutname = data_path + '2-do-ICA/' + p + '_nkg_part' + block + '_raw_sss_movecomp_tr_ica.fif'
	raw2.save(foutname)
	
