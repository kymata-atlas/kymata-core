#!/bin/bash
#

################
# PARAMETERS:

# Input/Output file path:

path='/imaging/at03/NKG_Data_Sets/DATASET_3-02_tactile_toes'

# Input/Output file stems:

parts=(\
 	'1' 
 	'2' 
	'3'
	'4'
 )

subjects=(\
        'meg15_0537_1'
	'meg15_0537_2' 
)

#####################
# SCRIPT BEGINS HERE:

nfiles=${#parts[*]}
lastfile=`expr $nfiles - 1`

nsubjects=${#subjects[*]}
lastsubj=`expr $nsubjects - 1`

# REPORT number of files to be processed:

for m in `seq 0 ${lastsubj}`
do

  echo " "
  echo "SUBJECT  ${subject_list[m]}: "

  for n in `seq 0 ${lastfile}`
  do
     

     echo " "
     echo " checking EEG channels for files nkg_${parts[n]}_raw.fif..."
     echo " "

     mne_check_eeg_locations --file ${path}/${subjects[m]}/block${parts[n]}_raw.fif --fix

	
  done # file loop
	
done # subject loop

echo " "
echo "DONE"
echo " "

# END OF SCRIPT
######################


