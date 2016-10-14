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
                    #'meg14_0173',
                    #'meg14_0178',
                    #'meg14_0436',
                    #'meg14_0191',
                    #'meg14_0193',
                    #'meg14_0213',
                    #'meg14_0219',
                    #'meg14_0226',
                    #'meg14_0230',
                    #'meg14_0231',
                    #'meg14_0239',
                    'meg14_0195'   
]


data_path = '/imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/'

block = '2'

for p in participants:

	# Set up data
	raw_fname = data_path + '1-preprosessing/sss/' + p + '_nkg_part' + block + '_raw_sss_movecomp.fif'
	raw = Raw(raw_fname, preload=True)
#	layout = read_layout('/imaging/at03/NKG_Code/Version5/layouts/easycapM1.lay')

	###############################################################################
	# 1) Fit ICA model using the FastICA algorithm
	
	# Other available choices are `infomax` or `extended-infomax`
	# We pass a float value between 0 and 1 to select n_components based on the
	# percentage of variance explained by the PCA components.

#	ica = ICA(n_components=0.95, method='fastica')

	new_ica_fname = data_path + '2-do-ICA/' + p + '_nkg_part' + block + '_raw_sss_movecomp_icaTEST.fif'
	ica = read_ica(new_ica_fname)

	picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=False, stim=False, exclude='bads')

#	ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))

	# maximum number of components to reject
#	n_max_eog = 1 #one each for HEOG and VEOG

	###############################################################################
	# 2) identify bad components by analyzing latent sources.

	title = 'Sources related to %s artifacts (red)'

	# detect EOG by correlation


#	eog_inds, scores = ica.find_bads_eog(raw)
	#ica.plot_scores(scores[1], exclude=eog_inds, title=title % 'veog')

	#show_picks = np.abs(scores[1]).argsort()[::-1][:5]

	#ica.plot_sources(raw, show_picks, title=title % 'veog')
        #ica.plot_components(eog_inds, title=title % 'veog', colorbar=True)
	##ica.plot_components(eog_inds, ch_type='eeg', layout=layout, title=title % 'veog', colorbar=True)

	#eog_inds = eog_inds[:n_max_eog]
	#ica.exclude += eog_inds

	###############################################################################
	# 3) Assess component selection and unmixing quality

	# estimate average artifact
	eog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5, picks=picks).average()
	#ica.plot_sources(eog_evoked, exclude=eog_inds)  # plot EOG sources + selection
	#ica.plot_overlay(eog_evoked, exclude=eog_inds)  # plot EOG cleaning

	ica.plot_sources(eog_evoked, exclude=ica.exclude)  # plot EOG sources + selection
	ica.plot_overlay(eog_evoked, exclude=ica.exclude)  # plot EOG cleaning

	# check the amplitudes do not change
	ica.plot_overlay(raw)  # EOG artifacts remain

	# save data
	#ica_fname = data_path + '2-do-ICA/' + p + '_nkg_part' + block + '_raw_sss_movecomp_icaTEST.fif'
	#ica.save(ica_fname)
	#plt.close('all')

	#raw_ica_fname = data_path + '2-do-ICA/' + p + '_nkg_part' + block + '_raw_sss_movecomp_icaTEST.fif'
	#raw = Raw(my_raw_fname, preload=True)
	ica.apply(raw, copy=False)
	raw_ica_fname = data_path + '2-do-ICA/' + p + '_nkg_part' + block + '_raw_sss_movecomp_ica.fif'
	raw.save(raw_ica_fname, overwrite=True)
	



