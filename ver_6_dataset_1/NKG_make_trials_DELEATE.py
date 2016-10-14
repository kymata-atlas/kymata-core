
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
data_path               = '/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory'
tmin, tmax              = -0.2, 1.8
numtrials               = 400
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
	raw_fname_part1 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part1_raw_sss_movecomp_tr.fif'
	raw_fname_part2 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part2_raw_sss_movecomp_tr.fif'


	#   	Read raw data
	raw = io.Raw(raw_fname_part1, preload=True)
	raw2 = io.Raw(raw_fname_part2)

	raw = io.concatenate_raws(raws=[raw, raw2], preload=True)
	raw.filter(l_freq=None, h_freq=60, picks=None)
	raw.notch_filter(50, picks=None, n_jobs=1)

	raw_events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)

	#	Correct for equiptment latency error
	raw_events = mne.event.shift_time_events(raw_events, [2], visual_delivery_latency, 1)
#	raw_events = mne.event.shift_time_events(raw_events, [3], audio_delivery_latency, 1)

	#	Extract visual events
	events_viz = mne.pick_events(raw_events, include=2)
	itemname = 1;	
	for item in events_viz:
		item[2] = itemname		#rename as '1,2,3 ...400' etc
		if itemname == 400:
			itemname = 1
		else :
			itemname = itemname+1

#	#	Extract audio events
#	events_audio_raw = mne.pick_events(raw_events, include=3)
#	events_audio = np.zeros((len(events_audio_raw)*400,3), dtype=int)
#	for i, item in enumerate(events_audio_raw):
#		for ii in xrange(400):
#			events_audio[((i*400)+(ii+1))-1][0] = item[0]+(ii*1000)
#			events_audio[((i*400)+(ii+1))-1][1] = 0
#			events_audio[((i*400)+(ii+1))-1][2] = ii+1
	

	##   	Plot raw data
	#fig = raw.plot(events=events)
	

	#	Denote picks
	include = [] # MISC05, trigger channels etc, if needed
	picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, exclude='bads')

#	for domain in ['audio', 'visual']: 
	for domain in ['visual']: 


		if domain =='audio':
			events = events_audio
		else:
			events = events_viz

		#	Extract trial instances ('epochs')
		epochs = mne.Epochs(raw, events, None, tmin, tmax, picks=picks,
		                   baseline=(None, None), reject=dict(eeg = eeg_thresh, grad = grad_thresh, mag = mag_thresh),
		                   preload=True)
	
		# 	Log which channels are worst 
		#dropfig = epochs.plot_drop_log(subject = p)
		#dropfig.savefig(data_path + '/3-sensor-data/logs/drop-log_' + p + '.jpg')
		
		global_droplog.append('[' + domain + ']' + p + ':' + str(epochs.drop_log_stats(epochs.drop_log)))
	
		#	Make and save trials as evoked data
		for i in range(1, numtrials+1):
			#evoked_one.plot() #(on top of each other)
			#evoked_one.plot_image() #(side-by-side)
			evoked = epochs[str(i)].average()  # average epochs and get an Evoked dataset.
			evoked.save(data_path + '/3-sensor-data/fif-out/'+ domain +'/' + p + '_item' + str(i) + '-ave.fif')

	# save grand average
	evoked_grandaverage = epochs.average() 
	evoked_grandaverage.save(data_path + '/3-sensor-data/fif-out/' + p + '-grandave.fif')	

	# save grand covs
	cov = mne.compute_covariance(epochs, tmin=None, tmax=None, method = 'auto', return_estimators=True)
	cov.save(data_path + '/3-sensor-data/covarience-files/' + p + '-auto-gcov.fif')



#	Save global droplog
f = open(data_path + '/3-sensor-data/logs/drop-log.txt', 'w')
print>>f,'Average drop rate for each participant\n'
for item in global_droplog:
	print>>f,item
f.close()


