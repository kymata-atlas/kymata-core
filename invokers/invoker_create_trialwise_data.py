from argparse import ArgumentParser
from logging import basicConfig, INFO
from pathlib import Path

from kymata.io.config import load_config, get_root_dir
from kymata.io.logging import log_message, date_format
from kymata.preproc.data_cleansing import create_trialwise_data


def main(config_filename: str):
    config = load_config(str(Path(Path(__file__).parent.parent, "dataset_config", config_filename)))

    create_trialwise_data(
        data_root_dir = get_root_dir(config),
        dataset_directory_name=config['dataset_directory_name'],
        list_of_participants=config['participants'],
        repetitions_per_runs=config['repetitions_per_runs'],
        stimulus_length=config['stimulus_lengths'],
        number_of_runs=config['number_of_runs'],
        latency_range=config["latency_range"],
    )


if __name__ == '__main__':
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    parser = ArgumentParser(description='Create Trialwise Data')
    parser.add_argument('--config', type=str, default="dataset4.yaml")
    args = parser.parse_args()
    main(config_filename=args.config)
