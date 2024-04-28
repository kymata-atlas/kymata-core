from argparse import ArgumentParser
from pathlib import Path

from kymata.io.yaml import load_config
from kymata.preproc.data_cleansing import create_trialwise_data


# noinspection DuplicatedCode
def main(config_filename: str):
    config = load_config(str(Path(Path(__file__).parent.parent, "dataset_config", config_filename)))

    create_trialwise_data(
        dataset_directory_name=config['dataset_directory_name'],
        list_of_participants=config['participants'],
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
    parser = ArgumentParser(description='Create Trialwise Data')
    parser.add_argument('--config', type=str, default="dataset4.yaml")
    args = parser.parse_args()
    main(config_filename=args.config)
