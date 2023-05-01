from colorama import Fore
from colorama import Style
import matplotlib.pyplot as plt
import mne
import os.path
import sklearn
import seaborn as sns
import numpy as np

import utils


def run_preprocessing(config: dict):

    list_of_participants = config['list_of_participants']
    number_of_runs = config['number_of_runs']
    remove_ECG = config['remove_ECG']
    skip_maxfilter_if_previous_runs_exist = config['skip_maxfilter_if_previous_runs_exist']
    remove_VEOH_and_HEOG = config['remove_VEOH_and_HEOG']
    automatic_bad_channel_detection_requested = config['automatic_bad_channel_detection_requested']

    '''Runs Preprocessing'''

    for participant in list_of_participants:

        for run in range(1, number_of_runs+1):

            # Preprocessing Participant and run info
            print(f"{Fore.GREEN}{Style.BRIGHT}Loading participant {participant} [{Fore.LIGHTYELLOW_EX}Run {str(run)}{Fore.GREEN}]...{Style.RESET_ALL}")

            # Load data
            print(f"{Fore.GREEN}{Style.BRIGHT}   Loading Raw data...{Style.RESET_ALL}")

            # set filename. (Use .fif.gz extension to use gzip to compress)
            saved_maxfiltered_filename = 'data/out/1_maxfiltered/' + participant + "_part" + str(run) + '_raw_sss.fif'

            if skip_maxfilter_if_previous_runs_exist and os.path.isfile(saved_maxfiltered_filename):
                raw_fif_data_sss_movecomp_tr = mne.io.Raw(saved_maxfiltered_filename, preload=True)

            else:
                raw_fif_data = mne.io.Raw("data/raw/" + participant + "_part" + str(run) + "_raw.fif", preload=True)
                head_pos = mne.chpi.read_head_pos("data/raw/" + participant + "_part" + str(run) + '_raw_hpi_movecomp.pos')

                # Rename any channels that require it, and their type
                recording_config = utils.load_recording_config('data/raw/' + participant + '_recording_config.yaml')
                ecg_and_eog_channel_name_and_type_overwrites = recording_config['ECG_and_EOG_channel_name_and_type_overwrites']

                # Set EOG and ECG types for clarity (normally shouldn't change anything, but we do it anyway)
                ecg_and_eog_channel_type_overwrites = {}
                for key, value in ecg_and_eog_channel_name_and_type_overwrites.items():
                    ecg_and_eog_channel_type_overwrites[key] = value["new_type"]
                raw_fif_data.set_channel_types(ecg_and_eog_channel_type_overwrites, verbose=None)

                # Set EOG and ECG names for clarity
                ecg_and_eog_channel_name_overwrites = {}
                for key, value in ecg_and_eog_channel_name_and_type_overwrites.items():
                    ecg_and_eog_channel_name_overwrites[key] = value["new_name"]
                raw_fif_data.rename_channels(ecg_and_eog_channel_name_overwrites, allow_duplicates=False)

                # Set bad channels (manual and automatic)
                print(f"{Fore.GREEN}{Style.BRIGHT}   Setting bad channels...{Style.RESET_ALL}")
                print(f"{Fore.GREEN}{Style.BRIGHT}   ...manual{Style.RESET_ALL}")

                raw_fif_data.info['bads'] = recording_config['bad_channels']

                if automatic_bad_channel_detection_requested:

                    print(f"{Fore.GREEN}{Style.BRIGHT}   ...automatic{Style.RESET_ALL}")
                    raw_fif_data = apply_automatic_bad_channel_detection()

                response = input(
                    f"{Fore.MAGENTA}{Style.BRIGHT}Would you like to see the raw data? Recommended if you want to confirm"
                    f" ECG, HEOG, VEOG are correct, and to mark further EEG bads (they will be saved directly) "
                    f" (y/n){Style.RESET_ALL}")
                if response == "y":
                    print(f"...Plotting Raw data.")
                    mne.viz.plot_raw(raw_fif_data, scalings='auto', block=True)
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
                    coord_frame='head',
                    st_correlation=0.980,
                    st_duration=10,
                    destination=(0, 0, 0.04),
                    verbose=True)

                raw_fif_data_sss_movecomp_tr.save(saved_maxfiltered_filename, fmt='short')

            response = input(f"{Fore.MAGENTA}{Style.BRIGHT}Would you like to see the SSS, movement compensated, raw data data? (y/n){Style.RESET_ALL}")
            if response == "y":
                print(f"...Plotting Raw data.")
                mne.viz.plot_raw(raw_fif_data_sss_movecomp_tr)
            else:
                print(f"[y] not pressed. Assuming you want to continue without looking at the raw data.")

            # Remove AC mainline from MEG

            print(f"{Fore.GREEN}{Style.BRIGHT}   Removing mains component (50Hz and harmonics) from MEG...{Style.RESET_ALL}")

            raw_fif_data_sss_movecomp_tr.compute_psd(tmax=1000000, fmax=500, average='mean').plot()

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

            raw_fif_data_sss_movecomp_tr.save('data/out/2_cleaned/' + participant + "_part" + str(run) + '_cleaned_raw.fif.gz', overwrite=True)

def create_trials(config:dict):
    """Create trials objects from the raw data files (still in sensor space)"""

    list_of_participants = config['list_of_participants']
    number_of_runs = config['number_of_runs']
    number_of_trials = config['number_of_trials']

    eeg_thresh = float(config['eeg_thresh'])
    grad_thresh = float(config['grad_thresh'])
    mag_thresh = float(config['mag_thresh'])

    visual_delivery_latency = config['visual_delivery_latency']
    audio_delivery_latency = config['audio_delivery_latency']

    tmin = config['tmin']
    tmax = config['tmax']

    global_droplog = []

    print(f"{Fore.GREEN}{Style.BRIGHT}Starting trials and {Style.RESET_ALL}")

    for p in list_of_participants:

        print(f"{Fore.GREEN}{Style.BRIGHT}...Concatenating trials{Style.RESET_ALL}")

        cleaned_raws = []

        for run in range(1, number_of_runs+1):

            raw_fname = 'data/out/2_cleaned/' + p + '_part' + str(run) + '_cleaned_raw.fif.gz'
            raw = mne.io.Raw(raw_fname, preload=True)
            cleaned_raws.append(raw)

        raw = mne.io.concatenate_raws(raws=cleaned_raws, preload=True)

        raw_events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)

        #	Correct for equiptment latency error
        raw_events = mne.event.shift_time_events(raw_events, [2], visual_delivery_latency, 1)
        raw_events = mne.event.shift_time_events(raw_events, [3], audio_delivery_latency, 1)

        print(f"{Fore.GREEN}{Style.BRIGHT}...finding visual events{Style.RESET_ALL}")

        #	Extract visual events
        visual_events = mne.pick_events(raw_events, include=2)
        trigger_name = 1;
        for trigger in visual_events:
            trigger[2] = trigger_name  # rename as '1,2,3 ...400' etc
            if trigger_name == 400:
                trigger_name = 1
            else:
                trigger_name = trigger_name + 1

        print(f"{Fore.GREEN}{Style.BRIGHT}...finding audio events{Style.RESET_ALL}")

        #	Extract audio events
        audio_events_raw = mne.pick_events(raw_events, include=3)
        audio_events = np.zeros((len(audio_events_raw) * 400, 3), dtype=int)
        for run, item in enumerate(audio_events_raw):
            for trial in range(1, number_of_trials + 1):
                audio_events[(trial+(number_of_trials * run)) - 1][0] = item[0] + (trial * 1000)
                audio_events[(trial+(number_of_trials * run)) - 1][1] = 0
                audio_events[(trial+(number_of_trials * run)) - 1][2] = trial

        #	Denote picks
        include = []  # MISC05, trigger channels etc, if needed
        picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, exclude='bads')

        print(f"{Fore.GREEN}{Style.BRIGHT}... extract and save evoked data{Style.RESET_ALL}")

        for input_stream in ['auditory']:

            if input_stream == 'auditory':
                events = audio_events
            else:
                events = visual_events

            #	Extract trial instances ('epochs')
            epochs = mne.Epochs(raw, events, None, tmin, tmax, picks=picks,
                                baseline=(None, None), reject=dict(eeg=eeg_thresh, grad=grad_thresh, mag=mag_thresh),
                                preload=True)

            # 	Log which channels are worst
            dropfig = epochs.plot_drop_log(subject = p)
            dropfig.savefig('data/out/3_evoked_sensor_data/logs/' + input_stream +'_drop-log_' + p + '.jpg')

            global_droplog.append('[' + input_stream + ']' + p + ':' + str(epochs.drop_log_stats(epochs.drop_log)))

            #	Make and save trials as evoked data
            for i in range(1, number_of_trials + 1):
                # evoked_one.plot() #(on top of each other)
                # evoked_one.plot_image() #(side-by-side)
                evoked = epochs[str(i)].average()  # average epochs and get an Evoked dataset.
                evoked.save('data/out/3_evoked_sensor_data/evoked_data/' + input_stream + '/' + p + '_item' + str(i) + '-ave.fif', overwrite=True)

        # save grand average
        print(f"{Fore.GREEN}{Style.BRIGHT}... save grand average{Style.RESET_ALL}")

        evoked_grandaverage = epochs.average()
        evoked_grandaverage.save('data/out/3_evoked_sensor_data/evoked_grand_average/' + p + '-grandave.fif', overwrite=True)

        # save grand covs
        #print(f"{Fore.GREEN}{Style.BRIGHT}... save grand covariance matrix{Style.RESET_ALL}")

        #cov = mne.compute_covariance(epochs, tmin=None, tmax=None, method='auto', return_estimators=True)
        #cov.save('data/out/3_evoked_sensor_data/evoked_data/covariance_grand_average/' + p + '-auto-gcov.fif')

    #	Save global droplog
    with open('data/out/3_evoked_sensor_data/logs/drop-log.txt', 'a') as file:
        file.write('Average drop rate for each participant\n')
        for item in global_droplog:
            file.write( item + '/n')

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