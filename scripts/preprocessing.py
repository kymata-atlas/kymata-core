from colorama import Fore
from colorama import Style

def run_preprocessing():
    '''Runs Preprocessing'''

     # set badchannels (manual and automatic)

    manual

    find_bad_channels_maxwell()

    # Check EEG locations

    NKG_mne_check_eeg_locations.sh (can this be done in MNE now?)
                                    
    Some versions of the Neuromag acquisition software did not copy the EEG channel location
    information properly from the Polhemus digitizer information data block to the EEG channel
    information records if the number of EEG channels exceeds 60. The purpose of mne_check_eeg_locations
    is to detect this problem and fix it, if requested. The command-line options are:

--file  <*name*>

Specify the measurement data file to be checked or modified.
--dig  <*name*>

Name of the file containing the Polhemus digitizer information. Default is the data file name.
--fix

By default mne_check_eeg_locations only checks for missing EEG locations (locations close to the origin). With –fix mne_check_eeg_locations reads the Polhemus data from the specified file and copies the EEG electrode location information to the channel information records in the measurement file. There is no harm running mne_check_eeg_locations on a data file even if the EEG channel locations were correct in the first place.

    # Apply SSS and movement compensation


    https://mne.tools/stable/auto_tutorials/preprocessing/60_maxwell_filtering_sss.html
    xxx = mne.preprocessing.maxwell_filter(XXX)

     maxfiltercmd = ['/neuro/bin/util/maxfilter -f ' [rawfname] ' -o ' [outfname] ,...
        ' -ctc /neuro/databases/ctc/ct_sparse.fif ',...
        ' -cal /neuro/databases/sss/sss_cal.dat ',...
        ' -linefreq 50 ',... % gets rid of mains frequency
        ' -autobad off ',...
        ' -trans ' [nameoffirstblock] ' ',... % transforms all blocks to same co-oridinates with respct to headpositions of the first block
        ' -st 4 ',... % SSS with ST
        ' -corr 0.980 ',... % SSS with ST
        ' -frame head ',...
        ' -origin 0 0 45 ',... %Manual sphere z-coordinate: 55 mm for low-landmarks, 45 mm for high landmarks
        ' -hpistep 200 ',... % movement compensation
        ' -hpisubt amp ',... % movement compensation
        ' -movecomp ',...% movement compensation
        ' -hp ' [rawfname] '_hpi_movecomp.pos ',... % movement compensation
        ' -format short ',...
        ' -v | tee ' [logfname]
        ];

    # Apply maxwell filtering to the empty room

    .maxwell_filter_prepare_emptyroom,
    .maxwell_filter

    # Remove AC power from MEG (mainline)

    meg_picks = mne.pick_types(raw.info, meg=True)
    freqs = (60, 120, 180, 240)
    raw_notch = raw.copy().notch_filter(freqs=freqs, picks=meg_picks)

    https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html

    #MEG and EEG channel interpolation

    MEG and bad channel interpolation method to RAW

    # Use common average reference, not the nose reference.

    https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html


    #remove very slow drift
    https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html

    # Remove ECG 

    preprocessing.

    # Remove VEOH, HEOG eyeblinks 

    if request of :
        preprocessing.
    else remove these trials with eyeblinks:
        xxx

    # Downsample if required
    https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html