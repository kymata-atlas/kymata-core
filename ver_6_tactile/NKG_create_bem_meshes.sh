#!/bin/bash
#

# setenv SUBJECTS_DIR /imaging/at03/NKG_Data_Sets/DATASET_1-01_visual-only/mne_subjects_dir/
# mne_setup_2.7.3_64bit
# setenv PATH /home/at03/MNE-2.7.4/bin:$PATH

SUBJECTS_DIR='/imaging/at03/NKG_Data_Sets/DATASET_3-01_visual-and-auditory/mne_subjects_dir/'    # root directory for MRI data

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

nsubjects=${#subjects[*]}
lastsubj=`expr $nsubjects - 1`


for m in `seq 0 ${lastsubj}`
do
        # creates surfaces necessary for BEM head models
        mne_watershed_bem --overwrite --subject ${subjects[m]}
        ln -s $SUBJECTS_DIR/${subjects[m]}/bem/watershed/${subjects[m]}'_inner_skull_surface' $SUBJECTS_DIR/${subjects[m]}/bem/inner_skull.surf
        ln -s $SUBJECTS_DIR/${subjects[m]}/bem/watershed/${subjects[m]}'_outer_skull_surface' $SUBJECTS_DIR/${subjects[m]}/bem/outer_skull.surf
        ln -s $SUBJECTS_DIR/${subjects[m]}/bem/watershed/${subjects[m]}'_outer_skin_surface'  $SUBJECTS_DIR/${subjects[m]}/bem/outer_skin.surf
        ln -s $SUBJECTS_DIR/${subjects[m]}/bem/watershed/${subjects[m]}'_brain_surface'       $SUBJECTS_DIR/${subjects[m]}/bem/brain_surface.surf
        # creates fiff-files for MNE describing MRI data
        mne_setup_mri --overwrite --subject ${subjects[m]}
        # create a source space from the cortical surface created in Freesurfer
        mne_setup_source_space --ico 5 --overwrite --subject ${subjects[m]} --cps
done
