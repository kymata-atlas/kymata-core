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

subjects=(\
        #'0045'
	#'0051'
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
	'0079'
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
  	echo " Creating labels for SUBJECT  ${subjects[m]}"
  	echo " "
  	echo " "
  	echo " "


	# Make labels


	cd ${path}${subjects[m]}/label/
	mkdir Destrieux_Atlas
	mkdir DK_Atlas
	mkdir DKT_Atlas

	cd Destrieux_Atlas
	mne_annot2labels --subject ${subjects[m]} --parc aparc.a2009s

	cd ../DK_Atlas
	mne_annot2labels --subject ${subjects[m]} --parc aparc

	cd ../DKT_Atlas
	mne_annot2labels --subject ${subjects[m]} --parc aparc.DKTatlas40



done # subject loop

echo " "
echo "DONE"
echo " "

# END OF SCRIPT
######################
