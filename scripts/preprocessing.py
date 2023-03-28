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
#     raw.filter(l_freq=None, h_freq=100, picks=None) (higher??)
#
#    # Remove ECG
#
#    preprocessing.
#
#    # Remove VEOH, HEOG eyeblinks
    print(f"{Fore.GREEN}{Style.BRIGHT}Starting ECG and EOG removal...{Style.RESET_ALL}")
    print(f"...Starting ECG and EOG removal.")
#
#    if request of :
#        preprocessing.
#    else remove these trials with eyeblinks:

        filt_raw = raw.copy().filter(l_freq=1., h_freq=None)

        ica = mne.preprocessing.ICA(n_components=30, method='picard', max_iter='auto', random_state=97)
        ica.fit(filt_raw)
        ica

        # remove ECG
        print(f"...Starting ECG removal.")

        ecg_evoked = mne.preprocessing.create_ecg_epochs(filt_raw).average()
        ecg_evoked.apply_baseline(baseline=(None, -0.2))
        ecg_evoked.plot_joint()

        # find which ICs match the EOG pattern
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        ica.exclude = eog_indices

        # barplot of ICA component "EOG match" scores
        ica.plot_scores(eog_scores)

        # plot diagnostics
        ica.plot_properties(raw, picks=eog_indices)

        # plot ICs applied to raw data, with EOG matches highlighted
        ica.plot_sources(raw, show_scrollbars=False)

        # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        ica.plot_sources(eog_evoked)

        # blinks
        ica.plot_overlay(raw, exclude=[0], picks='eeg')
        # heartbeats
        ica.plot_overlay(raw, exclude=[1], picks='mag')

        Apply it.

        # clean up memory before moving on
        del raw, ica, new_ica


        # remove EOG

        print(f"...Starting EOG removal.")

        eog_epochs = mne.preprocessing.create_eog_epochs(raw_fif_data_sss_movecomp_tr, baseline=(-0.5, -0.2))
        eog_epochs.plot_image(combine='mean')
        eog_epochs.average().plot_joint()

        eog_evoked = mne.preprocessing.create_eog_epochs(raw_fif_data_sss_movecomp_tr).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))
        eog_evoked.plot_joint()

        # note  seperate this into components!!!!!
        explained_var_ratio = ica.get_explained_variance_ratio(filt_raw, method='correlation',
                                                threshold='auto') # note can seperate this into components
        for channel_type, ratio in explained_var_ratio.items():
            print(
            f'Fraction of {channel_type} variance explained by all components: '
            f'{ratio}'
        )
        ica.exclude = []
        # find which ICs match the EOG pattern
        eog_indices, eog_scores = ica.find_bads_eog(filt_raw, method='correlation',
                                                threshold='auto')
        ica.exclude = eog_indices

        # barplot of ICA component "EOG match" scores
        ica.plot_scores(eog_scores)

        # plot diagnostics
        ica.plot_properties(raw, picks=eog_indices)

        # plot ICs applied to raw data, with EOG matches highlighted
        ica.plot_sources(raw, show_scrollbars=False)

        # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        ica.plot_sources(eog_evoked)

        # blinks
        ica.plot_overlay(raw, exclude=[0], picks='eeg')
        # heartbeats
        ica.plot_overlay(raw, exclude=[1], picks='mag')


        Apply it.

        # clean up memory before moving on
        del raw, ica, new_ica
#
#    # Downsample if required
#    https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html

'''
def create_trials(list_of_participants: [str],
                  input_stream: str):

    # Set Configs
    mne.set_config(xxx='MNE_STIM_CHANNEL', xxx='STI101')
    #
    # Set parameters
    audio_delivery_latency = 16;  # in milliseconds
    visual_delivery_latency = 34;  # in milliseconds
    tactile_delivery_latency = 0;  # in milliseconds
    data_path = '/imaging/at03/NKG_Code_output/Version5/DATASET_3-01_visual-and-auditory'
    tmin, tmax = -0.2, 1.8
    numtrials = 400

    eeg_thresh, grad_thresh, mag_thresh = 200e-6, 4000e-13, 4e-12


    # Inititialise global variables
    global_droplog = [];

    # Main Script

    for p in list_of_participants:

        #	Setup filenames
        raw_fname_part1 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part1_raw_sss_movecomp_tr.fif'
        raw_fname_part2 = data_path + '/1-preprosessing/sss/' + p + '_nkg_part2_raw_sss_movecomp_tr.fif'

        #   	Read raw data
        raw = io.Raw(raw_fname_part1, preload=True)
        raw2 = io.Raw(raw_fname_part2)

        raw = io.concatenate_raws(raws=[raw, raw2], preload=True)
        confirm no jump btewwn two (shoul dhave been sort aout by movecomp)?

        raw_events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)

        #	Correct for equiptment latency error
        raw_events = mne.event.shift_time_events(raw_events, [2], visual_delivery_latency, 1)
        raw_events = mne.event.shift_time_events(raw_events, [3], audio_delivery_latency, 1)

        #	Extract visual events
        events_viz = mne.pick_events(raw_events, include=2)
        itemname = 1;
        for item in events_viz:
            item[2] = itemname  # rename as '1,2,3 ...400' etc
            if itemname == 400:
                itemname = 1
            else:
                itemname = itemname + 1

        #	Extract audio events
        events_audio_raw = mne.pick_events(raw_events, include=3)
        events_audio = np.zeros((len(events_audio_raw) * 400, 3), dtype=int)
        for i, item in enumerate(events_audio_raw):
            for ii in xrange(400):
                events_audio[((i * 400) + (ii + 1)) - 1][0] = item[0] + (ii * 1000)
                events_audio[((i * 400) + (ii + 1)) - 1][1] = 0
                events_audio[((i * 400) + (ii + 1)) - 1][2] = ii + 1

        ##   	Plot raw data
        # fig = raw.plot(events=events)

        #	Denote picks
        include = []  # MISC05, trigger channels etc, if needed
        picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, exclude='bads')

        for domain in ['audio', 'visual']:

            if domain == 'audio':
                events = events_audio
            else:
                events = events_viz

            #	Extract trial instances ('epochs')
            epochs = mne.Epochs(raw, events, None, tmin, tmax, picks=picks,
                                baseline=(None, None), reject=dict(eeg=eeg_thresh, grad=grad_thresh, mag=mag_thresh),
                                preload=True)

            # 	Log which channels are worst
            # dropfig = epochs.plot_drop_log(subject = p)
            # dropfig.savefig(data_path + '/3-sensor-data/logs/drop-log_' + p + '.jpg')

            global_droplog.append('[' + domain + ']' + p + ':' + str(epochs.drop_log_stats(epochs.drop_log)))

            #	Make and save trials as evoked data
            for i in range(1, numtrials + 1):
                # evoked_one.plot() #(on top of each other)
                # evoked_one.plot_image() #(side-by-side)
                evoked = epochs[str(i)].average()  # average epochs and get an Evoked dataset.
                evoked.save(data_path + '/3-sensor-data/fif-out/' + domain + '/' + p + '_item' + str(i) + '-ave.fif')

        # save grand average
        evoked_grandaverage = epochs.average()
        evoked_grandaverage.save(data_path + '/3-sensor-data/fif-out/' + p + '-grandave.fif')

        # save grand covs
        cov = mne.compute_covariance(epochs, tmin=None, tmax=None, method='auto', return_estimators=True)
        cov.save(data_path + '/3-sensor-data/covariance-files/' + p + '-auto-gcov.fif')

    #	Save global droplog
    f = open(data_path + '/3-sensor-data/logs/drop-log.txt', 'w')
    print >> f, 'Average drop rate for each participant\n'
    for item in global_droplog:
        print >> f, item
    f.close()
'''