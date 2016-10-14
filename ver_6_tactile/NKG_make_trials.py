
print(__doc__)

import os
import numpy as np
import sys
sys.path.append('/imaging/local/software/mne_python/v0.9full')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.4')

import mne
from mne import io
from mne.io import concatenate_raws

###############################################################################
# Set Configs

mne.set_config('MNE_STIM_CHANNEL','STI101')

###############################################################################
# Set parameters
audio_delivery_latency  = 16;                               # in milliseconds
visual_delivery_latency = 34;                              	# in milliseconds
tactile_delivery_latency = 0;                              	# in milliseconds
data_path               = '/imaging/at03/NKG_Code_output/Version5/DATASET_3-02_tactile_toes'
tmin, tmax              = -0.2, 1.8
numtrials               = 400
eeg_thresh, grad_thresh, mag_thresh=200e-6, 4000e-13, 4e-12
participants = [
        #'meg15_0537_1'
	'meg15_0537_2' 
]

# Inititialise global variables
global_droplog = [];

# Main Script

for p in participants:

	#	Setup filenames
	raw_fname_part1 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part1_raw_sss_movecomp_tr.fif'
	raw_fname_part2 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part2_raw_sss_movecomp_tr.fif'
	raw_fname_part3 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part3_raw_sss_movecomp_tr.fif'
	raw_fname_part4 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part4_raw_sss_movecomp_tr.fif'


	#   	Read raw data
	raw = io.Raw(raw_fname_part1, preload=True)
	raw2 = io.Raw(raw_fname_part2)
	raw3 = io.Raw(raw_fname_part3)
	raw4 = io.Raw(raw_fname_part4)


	raw = io.concatenate_raws(raws=[raw, raw2, raw3, raw4], preload=True)
	raw.filter(l_freq=None, h_freq=20, picks=None)
	raw.notch_filter(50, picks=None, n_jobs=1)
	#raw.interpolate_bads()

	#raw_events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)
	#eve_fname = data_path + '/1-preprosessing/sss/' + p + '_nkg_eve.txt'
	#mne.write_events(eve_fname, raw_events)

	# new events
	eve_fname = data_path + '/1-preprosessing/sss/' + p + '_nkg_eve_net.txt'
	events = mne.read_events(eve_fname)

	##   	Plot raw data
	#fig = raw.plot(events=events)
	

	#	Denote picks
	include = [] # MISC05, trigger channels etc, if needed
	picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, exclude='bads')

	#	Extract trial instances ('epochs')
	epochs = mne.Epochs(raw, events, None, tmin, tmax, picks=picks,
	                   baseline=(None, None), reject=dict(eeg = eeg_thresh, grad = grad_thresh, mag = mag_thresh),
	                   preload=True)
	
	# 	Log which channels are worst 
	dropfig = epochs.plot_drop_log(subject = p)
	dropfig.savefig(data_path + '/3-sensor-data/logs/drop-log_' + p + '.jpg')
		
	global_droplog.append(p + ':' + str(epochs.drop_log_stats(epochs.drop_log)))
	
	#	Make and save trials as evoked data
	for i in range(1, numtrials+1):
		#evoked_one.plot() #(on top of each other)
		#evoked_one.plot_image() #(side-by-side)sss
		evoked = epochs[str(i)].average()  # average epochs and get an Evoked dataset.
		evoked.save(data_path + '/3-sensor-data/fif-out/' + p + '_item' + str(i) + '-ave.fif')

	# save grand average
	evoked_grandaverage = epochs.average() 
	evoked_grandaverage.save(data_path + '/3-sensor-data/fif-out/' + p + '-grandave.fif')	

	# save grand covs
	cov = mne.compute_covariance(epochs, tmin=None, tmax=None)
	cov.save(data_path + '/3-sensor-data/covarience-files/' + p + '-gcov.fif')



	#Save global droplog
f = open(data_path + '/3-sensor-data/logs/drop-log.txt', 'w')
print>>f,'Average drop rate for each participant\n'
for item in global_droplog:
	print>>f,item
f.close()


