#!/bin/sh
#
#before running script make sure you've typed:
# freesurfer_4.0.1
# mne_setup_2.7.0_64bit
# setenv $PATH
# setenv SUBJECTS_DIR .....




################
# PARAMETERS:

# Input/Output file path:

path="/imaging/at03/NKG_Data_Sets/DATASET_3-01_visual-and-auditory/mne_subjects_dir/"
datacode_path="/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/"

subjects=(\
        #'0045'
	'0051'
	#'0054'
	#'0055'
	#'0056'
	#'0058'
	#'0060'
	#'0065'
	#'0066'
	#'0070'
	#'0071'
	#'0072'
	#'0079'
	#'0081'
	#'0082'
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
  echo " Computing forward & inverse solution for SUBJECT  ${subjects[m]}"
  echo " "
  echo " "  echo " "


# Mke inverse oporator

mne_inverse_operator --csd --exclude $path${subjects[m]}/label/Destrieux_Atlas/Unknown-lh.label --exclude $path${subjects[m]}/label/Destrieux_Atlas/Unknown-rh.label --fwd ${datacode_path}3-sensor-data/forward-models/meg15_${subjects[m]}_ico-5-3L-fwd.fif --diagnoise --meg --eeg --loose 0.2 --senscov ${datacode_path}3-sensor-data/covarience-files/meg15_${subjects[m]}-gcov.fif --magreg 0.1 --gradreg 0.1 --eegreg 0.1 --inv ${datacode_path}3-sensor-data/inverse-operators/meg15_${subjects[m]}_ico-5-3L-loose02-diagnoise-nodepth-reg-inv-csd.fif


#MEG
#mne_do_inverse_operator --exclude $path${subjects[m]}/label/Destrieux_Atlas/Unknown-lh.label --exclude $path${subjects[m]}/label/Destrieux_Atlas/Unknown-rh.label --diagnoise --fwd /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/forward_models/meg14_${subjects[m]}_5-1L-MEG-fwd.fif --meg --loose 0.2 --senscov /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/averaged-noise-covarience-files/meg14_${subjects[m]}_gcov.fif --magreg 0.1 --gradreg 0.05 --eegreg 0.1 --inv /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/inverse-operators/meg14_${subjects[m]}_1L-diag-loose02-nodepth-reg-inv-MEGonly.fif

#EEG
#mne_do_inverse_operator --exclude $path${subjects[m]}/label/Destrieux_Atlas/Unknown-lh.label --exclude $path${subjects[m]}/label/Destrieux_Atlas/Unknown-rh.label --diagnoise --fwd /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/forward_models/meg14_${subjects[m]}_5-3L-EEG-fwd.fif --eeg --loose 0.2 --senscov /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/averaged-noise-covarience-files/meg14_${subjects[m]}_gcov.fif --magreg 0.05 --gradreg 0.05 --eegreg 0.1 --inv /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/inverse-operators/meg14_${subjects[m]}_3L-loose02-nodepth-reg-inv-EEGonly.fif




done # subject loop

echo " "
echo "DONE"
echo " "

# END OF SCRIPT
######################
