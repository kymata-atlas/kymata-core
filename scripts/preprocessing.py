from colorama import Fore
from colorama import Style
import matplotlib.pyplot as plt
import mne
import sklearn
import seaborn as sns
import numpy as np

import utils


def run_preprocessing(list_of_participants: str,
                      number_of_runs: int,
                      input_stream: str,
                      remove_ECG: bool,
                      remove_VEOH_and_HEOG: bool,
                      automatic_bad_channel_detection_requested: bool):

    '''Runs Preprocessing'''

    for participant in list_of_participants:

        for run in range(1, number_of_runs+1):

            # Preprocessing Participant and run info
            print(f"{Fore.GREEN}{Style.BRIGHT}Loading participant {participant} [{Fore.LIGHTYELLOW_EX}Run {str(run)}{Fore.GREEN}]...{Style.RESET_ALL}")

            # Load data
            print(f"{Fore.GREEN}{Style.BRIGHT}   Loading Raw data...{Style.RESET_ALL}")

            raw_fif_data = mne.io.Raw("data/raw/" + participant + "_part" + str(run) + "_raw.fif", preload=True)
            head_pos = mne.chpi.read_head_pos("data/raw/" + participant + "_part" + str(run) + '_raw_hpi_movecomp.pos')

            # Set bad channels (manual and automatic)
            print(f"{Fore.GREEN}{Style.BRIGHT}   Setting bad channels...{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{Style.BRIGHT}   ...manual{Style.RESET_ALL}")

            bads_config = utils.load_bad_channels('data/raw/' + participant + '_EMEG_bad_channels.yaml')
            raw_fif_data.info['bads'] = bads_config['bad_channels']

            if automatic_bad_channel_detection_requested:

                print(f"{Fore.GREEN}{Style.BRIGHT}   ...automatic{Style.RESET_ALL}")
                raw_fif_data = apply_automatic_bad_channel_detection()

            response = input(
                f"{Fore.MAGENTA}{Style.BRIGHT}Would you like to see the raw data? (y/n){Style.RESET_ALL}")
            if response == "y":
                print(f"...Plotting Raw data.")
                mne.viz.plot_raw(raw_fif_data, scalings='auto')
            else:
                print(f"...assuming you want to continue without looking at the raw data.")


            # plot EEG positions
            plot_eeg_sensor_positions(raw_fif_data)

            # Apply SSS and movement compensation
            print(f"{Fore.GREEN}{Style.BRIGHT}   Applying SSS and movement compensation...{Style.RESET_ALL}")

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
                print(f"[y] not pressed. Assuming you want to continue without looking at the raw data.")

            # Remove AC mainline from MEG

            print(f"{Fore.GREEN}{Style.BRIGHT}   Removing mains component (50Hz and harmonics) from MEG...{Style.RESET_ALL}")

            raw_fif_data_sss_movecomp_tr.compute_psd(tmax=1000000, fmax=500, average='mean').plot()

            #raw_fif_data_sss_movecomp_tr.save('data/out/" + participant + "_part" + str(run) + '_raw_sss.fif')
            #raw_fif_data_sss_movecomp_tr = mne.io.Raw('data/out/' + participant + "_part" + str(run) + '_raw_sss.fif', preload=True)

            # note that EEG and MEG do not have the same frequencies, so we remove them seperately
            meg_picks = mne.pick_types(raw_fif_data_sss_movecomp_tr.info, meg=True)
            meg_freqs = (50, 100, 150, 200, 250, 300, 350, 400, 293, 307, 314, 321, 328) # 293, 307, 314, 321, 328 are HPI coil frequencies
            raw_fif_data_sss_movecomp_tr = raw_fif_data_sss_movecomp_tr.notch_filter(freqs=meg_freqs, picks=meg_picks)

            eeg_picks = mne.pick_types(raw_fif_data_sss_movecomp_tr.info, eeg=True)
            eeg_freqs = (50, 150, 250, 300, 350, 400, 450, 293, 307, 314, 321, 328)
            raw_fif_data_sss_movecomp_tr = raw_fif_data_sss_movecomp_tr.notch_filter(freqs=eeg_freqs, picks=eeg_picks)


            # EEG channel interpolation
            print(f"{Fore.GREEN}{Style.BRIGHT}   Interpolating EEG...{Style.RESET_ALL}")

            # Channels marked as “bad” have been effectively repaired by SSS,
            # eliminating the need to perform MEG interpolation.

            raw_fif_data_sss_movecomp_tr = raw_fif_data_sss_movecomp_tr.interpolate_bads(reset_bads=True)

            # Use common average reference, not the nose reference.
            print(f"{Fore.GREEN}{Style.BRIGHT}   Use common average EEG reference...{Style.RESET_ALL}")

            raw_fif_data_sss_movecomp_tr = raw_fif_data_sss_movecomp_tr.set_eeg_reference(ref_channels='average')

            # remove very slow drift
            print(f"{Fore.GREEN}{Style.BRIGHT}   Removing slow drift...{Style.RESET_ALL}")
            raw_fif_data_sss_movecomp_tr = raw_fif_data_sss_movecomp_tr.filter(l_freq=0.1, h_freq=None, picks=None)

            # Remove ECG, VEOH and HEOG

            if remove_ECG or remove_VEOH_and_HEOG:

                # remove both frequencies faster than 100Hz and slow drift less than 1hz
                filt_raw = raw_fif_data_sss_movecomp_tr.copy().filter(l_freq=1., h_freq=100)

                ica = mne.preprocessing.ICA(n_components=30, method='fastica', max_iter='auto', random_state=97)
                ica.fit(filt_raw)

                explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
                for channel_type, ratio in explained_var_ratio.items():
                    print(
                        f'   Fraction of {channel_type} variance explained by all components: '
                        f'   {ratio}'
                    )

                ica.exclude = []

                if remove_ECG:
                    print(f"{Fore.GREEN}{Style.BRIGHT}   Starting ECG removal...{Style.RESET_ALL}")

                    ecg_evoked = mne.preprocessing.create_ecg_epochs(filt_raw).average()
                    ecg_evoked.apply_baseline(baseline=(None, -0.2))
                    ecg_evoked.plot_joint()

                    # find which ICs match the ECG pattern
                    ecg_indices, ecg_scores = ica.find_bads_ecg(filt_raw)
                    ica.exclude = ecg_indices

                    # plot ICs applied to raw data, with ECG matches highlighted
                    ica.plot_sources(filt_raw, show_scrollbars=True)

                    # barplot of ICA component "ECG match" scores
                    ica.plot_scores(ecg_scores)

                    # plot diagnostics
                    ica.plot_properties(filt_raw, picks=ecg_indices)

                    # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
                    ica.plot_sources(ecg_evoked)

                    # heartbeats
                    ica.plot_overlay(filt_raw, exclude=ica.exclude, picks='mag')

                if remove_VEOH_and_HEOG:

                    print(f"   ...Starting EOG removal.")

                    eog_evoked = mne.preprocessing.create_eog_epochs(filt_raw).average()
                    eog_evoked.apply_baseline(baseline=(None, -0.2))
                    eog_evoked.plot_joint()

                    # find which ICs match the EOG pattern
                    eog_indices, eog_scores = ica.find_bads_eog(filt_raw)
                    ica.exclude += eog_indices

                    # plot ICs applied to raw data, with EOG matches highlighted
                    ica.plot_sources(filt_raw, show_scrollbars=True)

                    # barplot of ICA component "EOG match" scores
                    ica.plot_scores(eog_scores)

                    # plot diagnostics
                    ica.plot_properties(filt_raw, picks=eog_indices)

                    # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
                    ica.plot_sources(eog_evoked)

                    # blinks
                    ica.plot_overlay(filt_raw, exclude=eog_indices, picks='eeg')

                ica.apply(raw_fif_data_sss_movecomp_tr)
                mne.viz.plot_raw(raw_fif_data_sss_movecomp_tr)

            raw_fif_data_sss_movecomp_tr.save('data/out/' + participant + "_part" + str(run) + '_cleaned_raw.fif')


def create_trials(list_of_participants: [str],
                  input_stream: str):
    """Create trials objects from the raw data files (still in sensor space)"""

    '''
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
def plot_eeg_sensor_positions(raw_fif: mne.io.Raw):
    '''Plot Sensor positions'''
    fig = plt.figure()
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection='3d')
    raw_fif.plot_sensors(ch_type='eeg', axes=ax2d)
    raw_fif.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d')
    ax3d.view_init(azim=70, elev=15)


def apply_automatic_bad_channel_detection(raw_fif_data: mne.io.Raw):
    '''Apply Automatic Bad Channel Detection'''
    raw_check = raw_fif_data.copy()

    fine_cal_file = 'data/cbu_specific_files/SSS/sss_cal.dat'
    crosstalk_file = 'data/cbu_specific_files/SSS/ct_sparse.fif'

    auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
        raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
        return_scores=True, verbose=True)
    print(auto_noisy_chs)
    print(auto_flat_chs)

    bads = raw_fif_data.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw_fif_data.info['bads'] = bads

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

    # Replace the word “noisy” with “flat”, and replace
    # vmin=np.nanmin(limits) with vmax=np.nanmax(limits) to print flat channels

    return raw_fif_data