import os.path

from colorama import Fore
import mne
from numpy import zeros, nanmin
from pandas import DataFrame, Index
from pathlib import Path

from kymata.io.cli import print_with_color, input_with_color
from kymata.io.yaml import load_config, modify_param_config


def run_first_pass_cleansing_and_maxwell_filtering(list_of_participants: list[str],
                      data_root_dir: str,
                      dataset_directory_name: str,
                      n_runs: int,
                      emeg_machine_used_to_record_data: str,
                      skip_maxfilter_if_previous_runs_exist: bool,
                      automatic_bad_channel_detection_requested: bool,
                      clean: bool,
                      ):

    for participant in list_of_participants:
        for run in range(1, n_runs + 1):

            # set filename. (Use .fif.gz extension to use gzip to compress)
            saved_maxfiltered_filename = data_root_dir + dataset_directory_name + '/intrim_preprocessing_files/1_maxfiltered/' + participant + "_run" + str(
                run) + '_raw_sss.fif'

            if skip_maxfilter_if_previous_runs_exist and os.path.isfile(saved_maxfiltered_filename):

                print_with_color(f"Skipping first pass filtering and Maxfiltering for {participant} [Run {str(run)}]...", Fore.GREEN)

            else:
                # Preprocessing Participant and run info
                print_with_color(f"Loading participant {participant} [Run {str(run)}]...", Fore.GREEN)

                # Load data
                print_with_color(f"   Loading Raw data...", Fore.GREEN)

                raw_fif_data = mne.io.Raw(
                    data_root_dir + dataset_directory_name + "/raw_emeg/" + participant + "/" + participant + "_run" + str(
                        run) + "_raw.fif", preload=True)

                # Rename any channels that require it, and their type
                recording_config = load_config(
                    data_root_dir + dataset_directory_name + '/raw_emeg/' + participant + "/" + participant + '_recording_config.yaml')
                ecg_and_eog_channel_name_and_type_overwrites = recording_config[
                    'ECG_and_EOG_channel_name_and_type_overwrites']

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

                # Set bad channels (manually)
                print_with_color(f"   Setting bad channels...", Fore.GREEN)
                print_with_color(f"   ...manual", Fore.GREEN)

                raw_fif_data.info['bads'] = recording_config['bad_channels']

                response = input_with_color(
                    f"Would you like to see the raw data? Recommended if you want to confirm"
                    f" ECG, HEOG, VEOG are correct, and to mark further EEG bads (they will be saved directly) "
                    f" (y/n)", Fore.MAGENTA)
                if response == "y":
                    print(f"...Plotting Raw data.")
                    mne.viz.plot_raw(raw_fif_data, scalings='auto', block=True)
                else:
                    print(f"...assuming you want to continue without looking at the raw data.")

                # Write back selected bad channels back to participant's config .yaml file
                modify_param_config(
                    data_root_dir + dataset_directory_name + '/raw_emeg/' + participant + "/" + participant + '_recording_config.yaml',
                    'bad_channels',
                    [str(item) for item in sorted(raw_fif_data.info['bads'])]
                )

                # Get the head positions
                chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw_fif_data)
                chpi_locs = mne.chpi.compute_chpi_locs(raw_fif_data.info, chpi_amplitudes)
                head_pos_data = mne.chpi.compute_head_pos(raw_fif_data.info, chpi_locs, verbose=True)

                print_with_color(f"   Removing CHPI ...", Fore.GREEN)

                # Remove hpi & line
                raw_fif_data = mne.chpi.filter_chpi(raw_fif_data, include_line=False)

                print_with_color(f"   Removing mains component (50Hz and harmonics) from MEG & EEG...", Fore.GREEN)

                raw_fif_data.compute_psd(tmax=1000000, fmax=500, average='mean').plot()

                # note that EEG and MEG do not have the same frequencies, so we remove them seperately
                meg_picks = mne.pick_types(raw_fif_data.info, meg=True)
                meg_freqs = (50, 100, 120, 150, 200, 240, 250, 360, 400, 450)
                raw_fif_data = raw_fif_data.notch_filter(freqs=meg_freqs, picks=meg_picks)

                eeg_picks = mne.pick_types(raw_fif_data.info, eeg=True)
                eeg_freqs = (50, 150, 250, 300, 350, 400, 450)
                raw_fif_data = raw_fif_data.notch_filter(freqs=eeg_freqs, picks=eeg_picks)

                raw_fif_data.compute_psd(tmax=1000000, fmax=500, average='mean').plot()

                if automatic_bad_channel_detection_requested:
                    print_with_color(f"   ...automatic", Fore.GREEN)
                    raw_fif_data = apply_automatic_bad_channel_detection(raw_fif_data, emeg_machine_used_to_record_data)

                # Apply SSS and movement compensation
                print_with_color(f"   Applying SSS and movement compensation...", Fore.GREEN)

                fine_cal_file = str(Path(Path(__file__).parent.parent.parent, 'kymata-toolbox-data', 'cbu_specific_files/SSS/sss_cal_' + emeg_machine_used_to_record_data + '.dat'))
                crosstalk_file = str(Path(Path(__file__).parent.parent.parent, 'kymata-toolbox-data', 'cbu_specific_files/SSS/ct_sparse_' + emeg_machine_used_to_record_data + '.fif'))
                
                if not clean:
                    mne.viz.plot_head_positions(
                        head_pos_data, mode='field', destination=raw_fif_data.info['dev_head_t'], info=raw_fif_data.info)

                raw_fif_data_sss_movecomp_tr = mne.preprocessing.maxwell_filter(
                    raw_fif_data,
                    cross_talk=crosstalk_file,
                    calibration=fine_cal_file,
                    head_pos=head_pos_data,
                    coord_frame='head',
                    st_correlation=0.980,
                    st_duration=10,
                    destination=(0, 0, 0.04),
                    verbose=True)

                raw_fif_data_sss_movecomp_tr.save(saved_maxfiltered_filename, fmt='short')

def run_second_pass_cleansing_and_EOG_removal(list_of_participants: list[str],
                                                      data_root_dir: str,
                                                      dataset_directory_name: str,
                                                      n_runs: int,
                                                      remove_ecg: bool,
                                                      remove_veoh_and_heog: bool,
                                                      skip_ica_if_previous_runs_exist: bool,
                                                      clean: bool,
                                                      ):
    for participant in list_of_participants:
        for run in range(1, n_runs + 1):

            saved_cleaned_filename = data_root_dir + dataset_directory_name + '/intrim_preprocessing_files/2_cleaned/' + participant + "_run" + str(
                run) + '_cleaned_raw.fif.gz'
            
            if skip_ica_if_previous_runs_exist and os.path.isfile(saved_cleaned_filename):

                print_with_color(f"Skipping second pass cleaning (ICA) for {participant} [Run {str(run)}]...", Fore.GREEN)

            else:

                # Preprocessing Participant and run info
                print_with_color(f"Loading participant {participant} [Run {str(run)}]...", Fore.GREEN)

                # set filename. (Use .fif.gz extension to use gzip to compress)
                saved_maxfiltered_filename = data_root_dir + dataset_directory_name + '/intrim_preprocessing_files/1_maxfiltered/' + participant + "_run" + str(
                    run) + '_raw_sss.fif'

                # Load data
                print_with_color(f"   Loading Raw data...", Fore.GREEN)

                raw_fif_data_sss_movecomp_tr = mne.io.Raw(saved_maxfiltered_filename, preload=True)

                if not clean:
                    response = input_with_color(
                        f"Would you like to see the SSS, movement compensated, raw data data? (y/n)",
                        Fore.MAGENTA)
                    if response == "y":
                        print(f"...Plotting Raw data.")
                        mne.viz.plot_raw(raw_fif_data_sss_movecomp_tr, block=True)
                    else:
                        print(f"[y] not pressed. Assuming you want to continue without looking at the raw data.")

                # EEG channel interpolation
                print_with_color(f"   Interpolating EEG...", Fore.GREEN)

                print("Bads channels: " + str(raw_fif_data_sss_movecomp_tr.info["bads"]))

                # Channels marked as “bad” have been effectively repaired by SSS,
                # eliminating the need to perform MEG interpolation.

                raw_fif_data_sss_movecomp_tr = raw_fif_data_sss_movecomp_tr.interpolate_bads(reset_bads=True,
                                                                                            mode='accurate')

                # Use common average reference, not the nose reference.
                print_with_color(f"   Use common average EEG reference...", Fore.GREEN)

                # raw_fif_data_sss_movecomp_tr = raw_fif_data_sss_movecomp_tr.set_eeg_reference(ref_channels='average')

                # remove very slow drift
                print_with_color(f"   Removing slow drift...", Fore.GREEN)
                raw_fif_data_sss_movecomp_tr = raw_fif_data_sss_movecomp_tr.filter(l_freq=0.1, h_freq=None, picks=None)

                # Remove ECG, VEOH and HEOG

                if remove_ecg or remove_veoh_and_heog:

                    # remove both frequencies faster than 40Hz and slow drift less than 1hz
                    filt_raw = raw_fif_data_sss_movecomp_tr.copy().filter(l_freq=1., h_freq=40)

                    ica = mne.preprocessing.ICA(n_components=30, method='fastica', max_iter='auto', random_state=97)
                    ica.fit(filt_raw)

                    explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
                    for channel_type, ratio in explained_var_ratio.items():
                        print(
                            f'   Fraction of {channel_type} variance explained by all components: '
                            f'   {ratio}'
                        )

                    ica.exclude = []

                    if remove_ecg:
                        _remove_ecg(filt_raw, ica)

                    if remove_veoh_and_heog:
                        _remove_veoh_and_heog(filt_raw, ica)

                    ica.apply(raw_fif_data_sss_movecomp_tr)

                    if not clean:
                        mne.viz.plot_raw(raw_fif_data_sss_movecomp_tr)

                raw_fif_data_sss_movecomp_tr.save(
                    data_root_dir + dataset_directory_name + '/intrim_preprocessing_files/2_cleaned/' + participant + "_run" + str(
                        run) + '_cleaned_raw.fif.gz',
                    overwrite=True)


def _remove_veoh_and_heog(filt_raw, ica):
    """
    Note: mutates `ica`.
    """
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


def _remove_ecg(filt_raw, ica):
    """
    Note: mutates `ica`.
    """
    print_with_color(f"   Starting ECG removal...", Fore.GREEN)
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


def create_trials(dataset_directory_name: str,
                  list_of_participants: list[str],
                  repetitions_per_runs: int,
                  number_of_runs: int,
                  number_of_trials: int,
                  input_streams: list[str],
                  eeg_thresh: float,
                  grad_thresh: float,
                  mag_thresh: float,
                  visual_delivery_latency: float,
                  audio_delivery_latency: float,
                  audio_delivery_shift_correction: float,
                  tmin: float,
                  tmax: float,
                  ):
    """Create trials objects from the raw data files (still in sensor space)."""

    global_droplog = []

    print_with_color(f"Starting trials and ", Fore.GREEN)

    for p in list_of_participants:

        print_with_color(f"...Concatenating trials", Fore.GREEN)

        cleaned_raws = []

        for run in range(1, number_of_runs + 1):
            raw_fname = '/imaging/projects/cbu/kymata/data/' + dataset_directory_name + '/intrim_preprocessing_files/2_cleaned/' + p + '_run' + str(run) + '_cleaned_raw.fif.gz'
            raw = mne.io.Raw(raw_fname, preload=True)
            cleaned_raws.append(raw)

        raw = mne.io.concatenate_raws(raws=cleaned_raws, preload=True)

        raw_events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)

        print_with_color(f"...finding visual events", Fore.GREEN)

        # Extract visual events
        visual_events = mne.pick_events(raw_events, include=[2, 3])

        # Correct for visual equiptment latency error
        visual_events = mne.event.shift_time_events(visual_events, [2, 3], visual_delivery_latency, 1)

        trigger_name = 1
        for trigger in visual_events:
            trigger[2] = trigger_name  # rename as '1,2,3 ...400' etc
            if trigger_name == 400:
                trigger_name = 1
            else:
                trigger_name = trigger_name + 1

        #  Test there are the correct number of events
        assert visual_events.shape[0] == repetitions_per_runs * number_of_runs * number_of_trials

        print_with_color(f"...finding audio events", Fore.GREEN)

        # Extract audio events
        audio_events_raw = mne.pick_events(raw_events, include=3)

        # Correct for audio latency error
        audio_events_raw = mne.event.shift_time_events(audio_events_raw, [3], audio_delivery_latency, 1)

        audio_events = zeros((len(audio_events_raw) * 400, 3), dtype=int)
        for run, item in enumerate(audio_events_raw):
            for trial in range(1, number_of_trials + 1):
                audio_events[(trial + (number_of_trials * run)) - 1][0] = item[0] + round(
                    (trial - 1) * (1000 + audio_delivery_shift_correction))
                audio_events[(trial + (number_of_trials * run)) - 1][1] = 0
                audio_events[(trial + (number_of_trials * run)) - 1][2] = trial

        #  Test there are the correct number of events
        assert audio_events.shape[0] == repetitions_per_runs * number_of_runs * number_of_trials

        # Denote picks
        include = []  # ['MISC006']  # MISC05, trigger channels etc, if needed
        picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, exclude='bads', include=include)

        print_with_color(f"... extract and save evoked data", Fore.GREEN)

        for input_stream in input_streams:

            if input_stream == 'auditory':
                events = audio_events
            else:
                events = visual_events

            # Extract trial instances ('epochs')
            epochs = mne.Epochs(raw, events, None, tmin, tmax, picks=picks,
                                baseline=(None, None), reject=dict(eeg=eeg_thresh, grad=grad_thresh, mag=mag_thresh),
                                preload=True)

            # 	Log which channels are worst
            dropfig = epochs.plot_drop_log(subject=p)
            dropfig.savefig(
                '/imaging/projects/cbu/kymata/data/' + dataset_directory_name + '/intrim_preprocessing_files/3_evoked_sensor_data/logs/' + input_stream + '_drop-log_' + p + '.jpg')

            global_droplog.append('[' + input_stream + ']' + p + ':' + str(epochs.drop_log_stats(epochs.drop_log)))

            # Make and save trials as evoked data
#            for i in range(1, number_of_trials + 1):
                # evoked_one.plot() #(on top of each other)
                # evoked_one.plot_image() #(side-by-side)

#                evoked = epochs[str(i)].average()  # average epochs and get an Evoked dataset.
#                evoked.save(
#                    'data/' + dataset_directory_name + '/intrim_preprocessing_files/3_evoked_sensor_data/evoked_data/' + input_stream + '/' + p + '_item' + str(
#                        i) + '-ave.fif', overwrite=True)

        # save grand average
#        print_with_color(f"... save grand average", Fore.GREEN)
#
#        evoked_grandaverage = epochs.average()
#        evoked_grandaverage.save(
#            'data/' + dataset_directory_name + '/intrim_preprocessing_files/3_evoked_sensor_data/evoked_grand_average/' + p + '-grandave.fif',
#            overwrite=True)

        # save grand covs
        print_with_color(f"... save grand covariance matrix", Fore.GREEN)

        cov = mne.compute_raw_covariance(raw, tmin=0, tmax=10, return_estimators=True)
        mne.write_cov('/imaging/projects/cbu/kymata/data/' + dataset_directory_name + '/intrim_preprocessing_files/3_evoked_sensor_data/covariance_grand_average/' + p + '-auto-cov.fif', cov)


# Save global droplog
#    with open('data/' + dataset_directory_name + '/intrim_preprocessing_files/3_evoked_sensor_data/logs/drop-log.txt', 'a') as file:
#        file.write('Average drop rate for each participant\n')
#        for item in global_droplog:
#            file.write(item + '/n')

    # Create average participant EMEG

#    print_with_color(f"... save participant average", Fore.GREEN)

#    for input_stream in input_streams:
#        for trial in range(1, number_of_trials + 1):
#            evokeds_list = []
#            for p in list_of_participants:
##                individual_evoked = mne.read_evokeds(
#                    'data/' + dataset_directory_name + '/intrim_preprocessing_files/3_evoked_sensor_data/evoked_data/' + input_stream + '/' + p + '_item' + str(
##                        trial) + '-ave.fif', condition=str(trial))
#                evokeds_list.append(individual_evoked)
#
##            average_participant_evoked = mne.combine_evoked(evokeds_list, weights="nave")
#            average_participant_evoked.save(
#                'data/' + dataset_directory_name + '/intrim_preprocessing_files/3_evoked_sensor_data/evoked_data/' + input_stream + '/item' + str(
#                    trial) + '-ave.fif', overwrite=True)

def apply_automatic_bad_channel_detection(raw_fif_data: mne.io.Raw, machine_used: str, plot: bool = True):
    """Apply Automatic Bad Channel Detection"""
    raw_check = raw_fif_data.copy()

    fine_cal_file = Path(Path(__file__).parent.parent,
                         'config', 'cbu_specific_files',
                         'sss_cal' + machine_used + '.dat')
    crosstalk_file = Path(Path(__file__).parent.parent,
                          'config', 'cbu_specific_files',
                          'ct_sparse' + machine_used + '.fif')

    auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
        raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
        return_scores=True, verbose=True)
    print(auto_noisy_chs)
    print(auto_flat_chs)

    bads = raw_fif_data.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw_fif_data.info['bads'] = bads

    if plot:
        _plot_bad_chans(auto_scores)

    return raw_fif_data


def _plot_bad_chans(auto_scores):
    import seaborn as sns
    from matplotlib import pyplot as plt

    # Only select the data for gradiometer channels.
    ch_type = 'grad'
    ch_subset = auto_scores['ch_types'] == ch_type
    ch_names = auto_scores['ch_names'][ch_subset]
    scores = auto_scores['scores_noisy'][ch_subset]
    limits = auto_scores['limits_noisy'][ch_subset]
    bins = auto_scores['bins']  # The windows that were evaluated.

    # We will label each segment by its start and stop time, with up to 3
    # digits before and 3 digits after the decimal place (1 ms precision).
    bin_labels = [f'{start:3.3f} – {stop:3.3f}'
                  for start, stop in bins]

    # We store the data in a Pandas DataFrame. The seaborn heatmap function
    # we will call below will then be able to automatically assign the correct
    # labels to all axes.
    data_to_plot = DataFrame(data=scores,
                             columns=Index(bin_labels, name='Time (s)'),
                             index=Index(ch_names, name='Channel'))

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
                vmin=nanmin(limits),  # bads in input data have NaN limits
                cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
    [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
     for x in range(1, len(bins))]
    ax[1].set_title('Scores > Limit', fontweight='bold')

    # The figure title should not overlap with the subplots.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Replace the word “noisy” with “flat”, and replace
    # vmin=nanmin(limits) with vmax=nanmax(limits) to print flat channels
