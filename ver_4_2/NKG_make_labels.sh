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

path='/imaging/at03/NKG_Data_Sets/DATASET_1-01_visual-only/mne_subjects_dir/'

subjects=(\
                    '0003'
                    '0006'
                    '0007'
                    '0009'
                    '0011'
                    '0013'
                    '0019'
                    '0020'
                    '0021'
                    '0022'
                    '0028'
                    '0039'
                    '0040'
                    '0041'
                    '0043'
                    '0045'
                    '0061'
                    '0063'
                    '0073'
                    '0075'   
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


	cd /imaging/at03/NKG_Data_Sets/VerbphraseMEG/nme_subject_dir/${subjects[m]}/label/
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
