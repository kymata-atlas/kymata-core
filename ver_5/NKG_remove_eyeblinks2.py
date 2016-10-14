print(__doc__)

## ICA for blink removal in MNE Python
##
## Based on the script of Caroline Whiting, June 2014
## NOTE: need to run ipython --matplotlib=qt to begin


import sys
sys.path.append('/imaging/local/software/mne_python/latest_v0.8')
sys.path.append('/imaging/local/software/python_packages/scikit-learn/v0.14.1')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.4')
import numpy as np
import mne
from mne.io import Raw
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.layouts import read_layout
from mne.preprocessing import read_ica


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
	'meg15_0068',
	'meg15_0070',
	'meg15_0071',
	'meg15_0072',
	'meg15_0079',
	'meg15_0081',
	'meg15_0082', 
	'meg15_0086' 
]


data_path = '/imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/'

block = '1'

for p in participants:

	# Set up data
	raw_fname = data_path + '1-preprosessing/sss/' + p + '_nkg_part' + block + '_raw_sss_movecomp_noEEGmainline.fif'
	raw = Raw(raw_fname, preload=True)
	#raw.filter(1, 100, n_jobs=1) # shouldn't be necassary
	layout = read_layout('/imaging/at03/NKG_Code/Version5/layouts/easycapM1.lay')
	picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, ecg=True, stim=False, exclude='bads')????????????????????????????????

	###############################################################################
	# 1) Fit ICA model using the FastICA algorithm
	
	# Other available choices are `infomax` or `extended-infomax`
	# We pass a float value between 0 and 1 to select n_components based on the
	# percentage of variance explained by the PCA components.

	ica = ICA(n_components=0.95, method='fastica')

	#new_ica_fname = data_path + '2-do-ICA/' + p + '_nkg_part' + block + '_raw_sss_movecomp_icaTEST.fif'
	#ica = read_ica(new_ica_fname)

	ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))

	# maximum number of components to reject
	n_max_veog, n_max_heog, n_max_ecg = 2,2,1 #one each for HEOG and VEOG

	###############################################################################
	# 2) identify bad components by analyzing latent sources.

	title = 'Sources related to %s artifacts (red)'

	# detect ECG by correlation

	# detect VEOG by correlation

	veog_inds, scores = ica.find_bads_veog(raw)
	ica.plot_scores(scores[1], exclude=veog_inds, title=title % 'veog')

	show_picks = np.abs(scores[1]).argsort()[::-1][:5]

	ica.plot_sources(raw, show_picks, title=title % 'veog')
        #ica.plot_components(eog_inds, title=title % 'veog', colorbar=True)
	##ica.plot_components(eog_inds, ch_type='eeg', layout=layout, title=title % 'veog', colorbar=True)

	eog_inds = eog_inds[:n_max_eog]
	ica.exclude += eog_inds

	###############################################################################
	# 3) Assess component selection and unmixing quality

	# estimate average artifact
	ecg_evoked = ecg_epochs.average()
	ica.plot_sources(ecg_evoked, exclude=ecg_inds)  # plot VEOG sources + selection
	ica.plot_overlay(ecg_evoked, exclude=ecg_inds)  # plot VEOG cleaning

	# estimate average artifact
	veog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5, picks=picks).average()
	ica.plot_sources(veog_evoked, exclude=eog_inds)  # plot VEOG sources + selection
	ica.plot_overlay(veog_evoked, exclude=eog_inds)  # plot VEOG cleaning

	# check the amplitudes do not change
	ica.plot_overlay(raw)  # VEOG artifacts remain

	ica.apply(raw, copy=False)
	raw_ica_fname = data_path + '2-do-ICA/' + p + '_nkg_part' + block + '_raw_sss_movecomp_ica.fif'
	raw.save(raw_ica_fname, overwrite=True)
	



