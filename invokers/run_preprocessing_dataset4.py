from pathlib import Path

import sys
sys.path.append('/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox')

from kymata.io.yaml import load_config
from kymata.preproc.pipeline import run_preprocessing, create_trials, create_trialwise_data


# noinspection DuplicatedCode
def main():
    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset4.yaml")))

    """create_trials(
        dataset_directory_name=config['dataset_directory_name'],
        list_of_participants=config['list_of_participants'],
        repetitions_per_runs=config['repetitions_per_runs'],
        number_of_runs=config['number_of_runs'],
        number_of_trials=config['number_of_trials'],
        input_streams=config['input_streams'],
        eeg_thresh=float(config['eeg_thresh']),
        grad_thresh=float(config['grad_thresh']),
        mag_thresh=float(config['mag_thresh']),
        visual_delivery_latency=config['visual_delivery_latency'],
        audio_delivery_latency=config['audio_delivery_latency'],
        audio_delivery_shift_correction=config['audio_delivery_shift_correction'],
        tmin=config['tmin'],
        tmax=config['tmax'],
    )"""

    """run_preprocessing(
        list_of_participants=config['list_of_participants'],
        dataset_directory_name=config['dataset_directory_name'],
        n_runs=config['number_of_runs'],
        emeg_machine_used_to_record_data=config['EMEG_machine_used_to_record_data'],
        remove_ecg=config['remove_ECG'],
        skip_maxfilter_if_previous_runs_exist=config['skip_maxfilter_if_previous_runs_exist'],
        remove_veoh_and_heog=config['remove_VEOH_and_HEOG'],
        automatic_bad_channel_detection_requested=config['automatic_bad_channel_detection_requested'],
    )"""

    create_trialwise_data(
        dataset_directory_name=config['dataset_directory_name'],
        list_of_participants=config['list_of_participants'],
        repetitions_per_runs=config['repetitions_per_runs'],
        number_of_runs=config['number_of_runs'],
        number_of_trials=config['number_of_trials'],
        input_streams=config['input_streams'],
        eeg_thresh=float(config['eeg_thresh']),
        grad_thresh=float(config['grad_thresh']),
        mag_thresh=float(config['mag_thresh']),
        visual_delivery_latency=config['visual_delivery_latency'],
        audio_delivery_latency=config['audio_delivery_latency'],
        audio_delivery_shift_correction=config['audio_delivery_shift_correction'],
        tmin=config['tmin'],
        tmax=config['tmax'],
    )


if __name__ == '__main__':
    main()
