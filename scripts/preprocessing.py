from colorama import Fore
from colorama import Style
import mne

def run_preprocessing(list_of_participants: str, input_stream: str):
    '''Runs Preprocessing'''

    # Load data
    print(f"{Fore.GREEN}{Style.BRIGHT}Loading Raw data...{Style.RESET_ALL}")

    raw_fif_data = mne.io.Raw("data/raw/meg15_0045_part2_raw.fif")
    head_pos = mne.chpi.read_head_pos('data/raw/meg15_0045_part2_raw_hpi_movecomp.pos')

    response = input(
        f"{Fore.MAGENTA}{Style.BRIGHT}Would you like to see the raw data? (y/n){Style.RESET_ALL}")
    if response == "y":
        print(f"...Plotting Raw data.")
        mne.viz.plot_raw(raw_fif_data)
    else:
        print(f"{Fore.RED}{Style.BRIGHT}[y] not pressed. Assuming you want to continue without looking at the raw data.{Style.RESET_ALL}")

    # Set bad channels (manual and automatic)
    print(f"{Fore.GREEN}{Style.BRIGHT}Setting bad channels...{Style.RESET_ALL}")
    '''
    print(f"{Fore.GREEN}{Style.BRIGHT}...manual{Style.RESET_ALL}")
    raw_fif_data.info['bads'] = xxxx()

    print(f"{Fore.GREEN}{Style.BRIGHT}...automatic{Style.RESET_ALL}")

    raw_check = raw_fif_data.copy()
    auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
        raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
        return_scores=True, verbose=True)
    print(auto_noisy_chs)  # we should find them!
    print(auto_flat_chs)  # none for this dataset

    bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw.info['bads'] = bads

    # Only select the data for gradiometer channels.
    ch_type = 'grad'
    ch_subset = auto_scores['ch_types'] == ch_type
    ch_names = auto_scores['ch_names'][ch_subset]
    scores = auto_scores['scores_noisy'][ch_subset]
    limits = auto_scores['limits_noisy'][ch_subset]
    bins = auto_scores['bins']  # The the windows that were evaluated.
    # We will label each segment by its start and stop time, with up to 3
    # digits before and 3 digits after the decimal place (1 ms precision).
    bin_labels = [f'{start:3.3f} – {stop:3.3f}'
                  for start, stop in bins]

    # We store the data in a Pandas DataFrame. The seaborn heatmap function
    # we will call below will then be able to automatically assign the correct
    # labels to all axes.
    data_to_plot = pd.DataFrame(data=scores,
                                columns=pd.Index(bin_labels, name='Time (s)'),
                                index=pd.Index(ch_names, name='Channel'))

    # First, plot the "raw" scores.
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Automated noisy channel detection: {ch_type}',
                 fontsize=16, fontweight='bold')
    sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
                ax=ax[0])
    [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
     for x in range(1, len(bins))]
    ax[0].set_title('All Scores', fontweight='bold')

    # Now, adjust the color range to highlight segments that exceeded the limit.
    sns.heatmap(data=data_to_plot,
                vmin=np.nanmin(limits),  # bads in input data have NaN limits
                cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
    [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
     for x in range(1, len(bins))]
    ax[1].set_title('Scores > Limit', fontweight='bold')

    # The figure title should not overlap with the subplots.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    You can use the very same code as above to produce figures for flat channel
        detection.Simply replace the word “noisy” with “flat”, and replace
        vmin=np.nanmin(limits) with vmax=np.nanmax(limits).'''

    # Apply SSS and movement compensation
    print(f"{Fore.GREEN}{Style.BRIGHT}Applying SSS and movement compensation...{Style.RESET_ALL}")

    fine_cal_file = 'data/cbu_specific_files/SSS/sss_cal.dat'
    crosstalk_file = 'data/cbu_specific_files/SSS/ct_sparse.fif'

    mne.viz.plot_head_positions(
        head_pos, mode='field', destination=raw_fif_data.info['dev_head_t'], info=raw_fif_data.info)

    raw_fif_data_sss_movecomp_tr = mne.preprocessing.maxwell_filter(
                                    raw_fif_data,
                                    cross_talk=crosstalk_file,
                                    calibration=fine_cal_file,
                                    head_pos=head_pos,
                                    st_correlation=0.980,
                                    st_duration=10,
                                    destination=(0, 0, 0.04),
                                    verbose=True)

    response = input(
        f"{Fore.MAGENTA}{Style.BRIGHT}Would you like to see the SSS, movement compensated, raw data data? (y/n){Style.RESET_ALL}")
    if response == "y":
        print(f"...Plotting Raw data.")
        mne.viz.plot_raw(raw_fif_data_sss_movecomp_tr)
    else:
        print(f"{Fore.RED}{Style.BRIGHT}[y] not pressed. Assuming you want to continue without looking at the raw data.{Style.RESET_ALL}")

#    # Remove AC mainline from MEG
#
#    meg_picks = mne.pick_types(raw.info, meg=True)
#    freqs = (50, 100, 150, 200)
#    raw_notch = raw.copy().notch_filter(freqs=freqs, picks=meg_picks)
#
#    https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
#
#    #MEG and EEG channel interpolation
#
# Notice that channels marked as “bad” have been effectively repaired by SSS, eliminating the need to perform interpolation. The heartbeat artifact has also been substantially reduced.
#
#    Compute interpolation (also works with Raw and Epochs objects)
#
# evoked_interp = evoked.copy().interpolate_bads(reset_bads=True)
# evoked_interp.plot(exclude=[], picks=('grad', 'eeg'))
#
#    # Use common average reference, not the nose reference.
#
#    https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html
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