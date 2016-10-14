#!/bin/sh


path='/imaging/at03/NKG_Data_Sets/DATASET_3-01_visual-and-auditory'

subjects=(\
	'0045'
	'0051'
	'0054'
	'0055'
	'0056'
	'0058'
	'0060'
	'0065'
	'0066'
	'0070'
	'0071'
	'0072'
	'0079'
	'0081'
	'0082'
)

#####################
# SCRIPT BEGINS HERE:

nsubjects=${#subjects[*]}
lastsubj=`expr $nsubjects - 1`


# REPORT number of files to be processed:

for m in `seq 0 ${lastsubj}`
do
  	echo " "
  	echo " "
  	echo " Marking bad channels for SUBJECT  ${subjects[m]}"
  	echo " "
  	echo " "
  	echo " "


	# Mark bad channels

	mne_mark_bad_channels --bad ${path}/meg15_${subjects[m]}/altEEGbads/EEG_bad_channels_fif_names.txt /imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/1-preprosessing/sss/meg15_${subjects[m]}_nkg_part1_raw_sss_movecomp_EEGmainline.fif /imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/1-preprosessing/sss/meg15_${subjects[m]}_nkg_part2_raw_sss_movecomp_EEGmainline.fif 


done #subject loop
