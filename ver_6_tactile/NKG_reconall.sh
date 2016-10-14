#!/bin/sh


path='/imaging/at03/NKG_Data_Sets/DATASET_3-01_visual-and-auditory'

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
  	echo " Recon-all for SUBJECT  ${subjects[m]}"
  	echo " "
  	echo " "
  	echo " "


	# Do recon-all

	recon-all -subjid ${subjects[m]} -autorecon2

done #subject loop
