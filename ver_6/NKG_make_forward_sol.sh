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

code_output_path="/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory/"
datasets_path="/imaging/at03/NKG_Data_Sets/DATASET_3-01_visual-and-auditory/"
mne_sub="/imaging/at03/NKG_Data_Sets/DATASET_3-01_visual-and-auditory/mne_subjects_dir/"

subjects=(\
        '0045'
	#'0051'
	#'0054'
	'0055'
	'0056'
	'0058'
	'0060'
	#'0065'
	'0066'
	'0070'
	'0071'
	'0072'
	'0079'
	'0081'
	'0082'
)

subjectsX=(\
        '0045'
	#'0173'
	#'0230'
	'0055'
	'0056'
	'0058'
	'0060'
	#'0178'
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
  echo " Computing forward & inverse solution for SUBJECT  ${subjects[m]}"
  echo " "
  echo " "
  echo " "

## creat bem-sol.fif (only needs to be done once)
# this only needs to be ico 4 - 5 will not work!!! https://mail.nmr.mgh.harvard.edu/pipermail//mne_analysis/2012-June/001102.html

#mne_setup_forward_model --overwite --subject ${subjects[m]} --surf --ico 4

#create forward solution 3 layers (both MEG & EEG, and EEg only). 'meas' is only suplying the sesnor locations and orianations.

mne_do_forward_solution --subject ${subjects[m]} --mindist 5 --ico 5 --bem ${mne_sub}${subjects[m]}/bem/${subjectsX[m]}-5120-5120-5120-bem-sol.fif --src ${mne_sub}${subjects[m]}/bem/${subjectsX[m]}-ico-5-src.fif --meas ${code_output_path}3-sensor-data/fif-out/'meg15_'${subjects[m]}-grandave.fif --fwd ${code_output_path}3-sensor-data/forward-models/meg15_${subjects[m]}_ico-5-3L-fwd.fif

#eegonly

#mne_do_forward_solution --subject ${subjects[m]} --mindist 5 --spacing 5 --eegonly --bem ${path}${subjects[m]}/bem/${subjects[m]}-5120-5120-5120-bem-sol.fif --src ${path}${subjects[m]}/bem/${subjects[m]}-5-src.fif --meas /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/fif-out-averaged/'meg14_'${subjects[m]}-grandave.fif --fwd /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/forward_models/meg14_${subjects[m]}_5-3L-EEG-fwd.fif


#Megonly - (forward solution  1 layer)

#mne_do_forward_solution --subject ${subjects[m]} --mindist 5 --spacing 5 --megonly --bem ${path}${subjects[m]}/bem/${subjects[m]}-5120-bem-sol.fif --src ${path}${subjects[m]}/bem/${subjects[m]}-5-src.fif --meas /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/fif-out-averaged/'meg14_'${subjects[m]}-grandave.fif --fwd /imaging/at03/NKG_Code_output/Version5/DATASET_1-01_visual-only/3-sensor-data/forward_models/meg14_${subjects[m]}_5-3L-MEG-fwd.fif



done # subject loop

echo " "
echo "DONE"
echo " "

# END OF SCRIPT
######################
