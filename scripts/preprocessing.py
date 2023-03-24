from colorama import Fore
from colorama import Style
import mne

def run_preprocessing(list_of_participants: str, input_stream: str):
    '''Runs Preprocessing'''

    # Load data
    print(f"{Fore.GREEN}{Style.BRIGHT}Loading Raw data...{Style.RESET_ALL}")

    raw_fif_data = mne.io.Raw("data/raw/meg15_0045_part1_raw.fif")

    response = input(
        f"{Fore.MAGENTA}{Style.BRIGHT}Would you like to see the raw data? (y/n){Style.RESET_ALL}")
    if response == "y":
        print(f"...Plotting Raw data.")
        mne.viz.plot_raw(raw_fif_data)
    else:
        print(f"{Fore.RED}{Style.BRIGHT}[y] not pressed. Assuming you want to continue without looking at the raw data.{Style.RESET_ALL}")

    # Set bad channels (manual and automatic)
    print(f"{Fore.GREEN}{Style.BRIGHT}Setting bad channels...{Style.RESET_ALL}")

#    manual

#    find_bad_channels_maxwell(), ask for confirmation.

    # Apply SSS and movement compensation

#    https://mne.tools/stable/auto_tutorials/preprocessing/60_maxwell_filtering_sss.html
#    xxx = mne.preprocessing.maxwell_filter(XXX)

#     maxfiltercmd = ['/neuro/bin/util/maxfilter -f ' [rawfname] ' -o ' [outfname] ,...
#        ' -ctc /neuro/databases/ctc/ct_sparse.fif ',...
#        ' -cal /neuro/databases/sss/sss_cal.dat ',...
#        ' -linefreq 50 ',... % gets rid of mains frequency
#        ' -autobad off ',...
#        ' -trans ' [nameoffirstblock] ' ',... % transforms all blocks to same co-oridinates with respct to headpositions of the first block
#        ' -st 4 ',... % SSS with ST
#        ' -corr 0.980 ',... % SSS with ST
#        ' -frame head ',...
#        ' -origin 0 0 45 ',... %Manual sphere z-coordinate: 55 mm for low-landmarks, 45 mm for high landmarks
#        ' -hpistep 200 ',... % movement compensation
#        ' -hpisubt amp ',... % movement compensation
#        ' -movecomp ',...% movement compensation
#        ' -hp ' [rawfname] '_hpi_movecomp.pos ',... % movement compensation
#        ' -format short ',...
#        ' -v | tee ' [logfname]
#        ];
#
#    # Apply maxwell filtering to the empty room
#
#    .maxwell_filter_prepare_emptyroom,
#    .maxwell_filter
#
#    # Remove AC mainline from MEG
#
#    meg_picks = mne.pick_types(raw.info, meg=True)
#    freqs = (60, 120, 180, 240)
#    raw_notch = raw.copy().notch_filter(freqs=freqs, picks=meg_picks)
#
#    https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
#
#    #MEG and EEG channel interpolation
#
#    MEG and bad channel interpolation method to RAW
#
#    # Use common average reference, not the nose reference.
#
#    https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html
#
#
#    #remove very slow drift
#    https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
#
#    # Remove ECG
#
#    preprocessing.
#
#    # Remove VEOH, HEOG eyeblinks
#
#    if request of :
#        preprocessing.
#    else remove these trials with eyeblinks:
#        xxx
#
#    # Downsample if required
#    https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html