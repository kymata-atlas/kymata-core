
print(__doc__)

import os
import numpy as np
import sys
sys.path.append('/imaging/local/software/mne_python/latest_v0.8')
sys.path.append('/imaging/local/software/python_packages/scikit-learn/v0.14.1')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.4')

import mne
from mne import io
from mne.io import concatenate_raws


###############################################################################
# Set parameters
data_path = '/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory'
tmin, tmax = -0.2, 1.8
numtrials=400
eeg_thresh, grad_thresh, mag_thresh=200e-6, 4000e-13, 4e-12
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
	'meg15_0082' 
]

# Inititialise global variables
global_droplog = [];

# Main Script

for p in participants:

	#	Setup filenames
	raw_fname_part1 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part1_raw_sss_movecomp_EEGmainline.fif'
	raw_fname_part2 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part2_raw_sss_movecomp_EEGmainline.fif'
	event_fname_part1 = data_path + '/3-sensor-data/event-files/audio/' + p + '_part1.eve'
	event_fname_part2 = data_path + '/3-sensor-data/event-files/audio/' + p + '_part2.eve'


	#   	Read raw data
	raw = io.Raw(raw_fname_part1, preload=True)
	#raw2 = io.Raw(raw_fname_part2)
	events = mne.read_events(event_fname_part1)
	#events2 = mne.read_events(event_fname_part2)

	#raw, events = io.concatenate_raws(raws=[raw, raw2], preload=True, events_list=[events, events2])
	raw.filter(l_freq=None, h_freq=45, picks=None)

	##   	Plot raw data
	#fig = raw.plot(events=events)
	

	#	Denote picks
	#include = ['MISC005'] # No MISC05, Trigger channels etc
	picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, exclude='bads')

	#	Extract trial instances ('epochs')
	epochs = mne.Epochs(raw, events, None, tmin, tmax, picks=picks,
	                   baseline=(None, None), reject=dict(eeg = eeg_thresh, grad = grad_thresh, mag = mag_thresh),
	                   preload=True)

	# 	Log which channels are worst 
	#dropfig = epochs.plot_drop_log(subject = p)
	#dropfig.savefig(data_path + '/3-sensor-data/logs/drop-log_' + p + '.jpg')
	
	global_droplog.append(p + ':' + str(epochs.drop_log_stats(epochs.drop_log)))

	#	Make and save trials as evoked data
	for i in range(1, numtrials+1):
		#evoked_one.plot() #(on top of each other)
		#evoked_one.plot_image() #(side-by-side)
		evoked = epochs[str(i)].average()  # average epochs and get an Evoked dataset.
		evoked.save(data_path + '/3-sensor-data/fif-out/' + p + '_item' + str(i) + '-ave.fif')

	# save grand average
	#evoked_grandaverage = epochs.average() 
	#evoked_grandaverage.save(data_path + '/3-sensor-data/fif-out/' + p + '-grandave.fif')	

	# save grand covs
	#cov = mne.compute_covariance(epochs, tmin=None, tmax=None) # but see to-do sheet for other possibilites to experiment with	
	#cov.save(data_path + '/3-sensor-data/covarience-files/' + p + '-gcov.fif')	



#	Save global droplog
f = open(data_path + '/3-sensor-data/logs/drop-log.txt', 'w')
print>>f,'Average drop rate for each participant\n'
for item in global_droplog:
	print>>f,item
f.close()
