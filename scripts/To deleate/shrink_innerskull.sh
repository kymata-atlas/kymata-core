#
#       Shrinks skull when boundaries cross http://imaging.mrc-cbu.cam.ac.uk/meg/AnalyzingData/MNE_ForwardSolution_ShrinkSkull
#	to run, run: 	[$] exec bash					(needs to be in bash to work)
#	then		[$] ./shrink_inner_skull.sh --subject ${SUBJECT} --overwrite
#	then   		deleate /watershed in /bem/, rename /watershed3 to /watershed
#	and then (which might alread exist) [$] ln -s  ${MRI_path}/${SUBJECT}/bem/watershed/${SUBJECT}_inner_skull_surface  ${MRI_path}/${SUBJECT}/bem/inner_skull.surf
#
#
#       FreeSurfer
#
#       Copyright 2006
#
#       Matti Hamalainen
#       Athinoula A. Martinos Center for Biomedical Imaging
#       Massachusetts General Hospital
#       Charlestown, MA, USA
#
#       $Header: /space/orsay/8/users/msh/CVS/CVS-MSH/MNE/mne_scripts/mne_watershed_bem,v 1.7 2008/10/23 21:41:45 msh Exp $
#       $log$
#
cleanup ()
{
        echo "Temporary files removed."
}
usage ()
{
        echo "usage: $0 [options]"
        echo 
        echo "     --overwrite             (to write over existing files)"
        echo "     --subject subject       (defaults to SUBJECT environment variable)"
        echo "     --volume  name          (defaults to T1)"
        echo "     --atlas                 specify the --atlas option for mri_watershed"
        echo
        echo "Minimal invocation:"
        echo
        echo "$0                      (SUBJECT environment variable set)"
        echo "$0 --subject subject    (define subject on the command line)"
        echo 
}
#
if [ ! "$FREESURFER_HOME" ] ; then 
    echo "The FreeSurfer environment needs to be set up for this script"
    exit 1
fi
#
if [ ! "$SUBJECTS_DIR" ]
then
        echo "The environment variable SUBJECTS_DIR should be set"
        exit 1
fi
if [ ! "$MNE_ROOT" ]
then
    echo "MNE_ROOT environment variable is not set"
    exit 1
fi
force=false
atlas=
volume=T1
#
#       Parse the options
#
while [ $# -gt 0 ]
do
        case "$1" in  
        --subject) 
                shift
                if [ $# -eq 0 ]
                then
                        echo "--subject: argument required."
                        exit 1
                else
                        export SUBJECT=$1
                fi
                ;;
        --volume) 
                shift
                if [ $# -eq 0 ]
                then
                        echo "--volume: argument required."
                        exit 1
                else
                        export volume=$1
                fi
                ;;
        --overwrite)
                force=true
                ;;
        --atlas)
                atlas=-atlas
                ;;
        --help)
                usage
                exit 1
                ;;
        esac

        shift
done
#
#       Check everything is alright
#
if [ ! "$SUBJECT" ]
then
        usage
        exit 1
fi
#
subject_dir=$SUBJECTS_DIR/$SUBJECT
mri_dir=$subject_dir/mri
T1_dir=$mri_dir/$volume
T1_mgz=$mri_dir/$volume.mgz
bem_dir=$subject_dir/bem
ws_dir=$subject_dir/bem/watershed3
#
if [ ! -d $subject_dir ]
then 
        echo "Could not find the MRI data directory $subject_dir"
        exit 1
fi
if [ ! -d $bem_dir ]
then 
    mkdir -p $bem_dir
    if [ $? -ne 0 ]
    then
        echo "Could not create the model directory $bem_dir"
        exit 1
    fi
fi
if [ ! -d $T1_dir -a ! -f $T1_mgz ]
then 
    echo "Could not find the MRI data"
    exit 1
fi
if [ -d $ws_dir ]
then
        if [ $force = "false" ]
        then
                echo "$ws_dir already exists. Use the --overwrite option to recreate it"
                exit 1
        else
                rm -rf $ws_dir
                if [ $? -ne 0 ]
                then 
                        echo "Could not remove $ws_dir"
                        exit 1
                fi
        fi
fi
#
#       Report
#
echo 
echo "Running mri_watershed for BEM segmentation with the following parameters"
echo
echo "SUBJECTS_DIR = $SUBJECTS_DIR"
echo "Subject      = $SUBJECT"
echo "Result dir   = $ws_dir"
echo
mkdir -p $ws_dir/ws
if [ $? -ne 0 ]
then 
    echo "Could not create the destination directories"
    exit 1
fi
cleanup
if [ -f $T1_mgz ]
then
    mri_watershed $atlas -useSRAS -shk_br_surf 2 int_h -surf $ws_dir/$SUBJECT $T1_mgz $ws_dir/ws
else
    mri_watershed $atlas -useSRAS -shk_br_surf 2 int_h -surf $ws_dir/$SUBJECT $T1_dir $ws_dir/ws
fi    
if [ $? -ne 0 ]
then
    exit 1
fi
#
cd $bem_dir
rm -f $SUBJECT-head.fif
mne_surf2bem --surf $ws_dir/"$SUBJECT"_outer_skin_surface --id 4 --fif $SUBJECT-head.fif
echo "Created $bem_dir/$SUBJECT-head.fif"
echo
echo "Complete."
echo
exit 0
