from logging import basicConfig, INFO
from pathlib import Path

from kymata.io.config import load_config, get_root_dir
from kymata.io.logging import log_message, date_format
from kymata.preproc.data_cleansing import (
    run_first_pass_cleansing_and_maxwell_filtering,
    run_second_pass_cleansing_and_eog_removal,
)


# noinspection DuplicatedCode
def main(config_filename: str):
    config = load_config(str(Path(Path(__file__).parent.parent, "dataset_config", config_filename)))

    data_root_dir = get_root_dir(config)

    run_first_pass_cleansing_and_maxwell_filtering(
        data_root_dir=data_root_dir,
        list_of_participants=config["participants"],
        dataset_directory_name=config["dataset_directory_name"],
        n_runs=config["number_of_runs"],
        emeg_machine_used_to_record_data=config["emeg_machine_used_to_record_data"],
        skip_maxfilter_if_previous_runs_exist=config["skip_maxfilter_if_previous_runs_exist"],
        automatic_bad_channel_detection_requested=config["automatic_bad_channel_detection_requested"],
        supress_excessive_plots_and_prompts=config["supress_excessive_plots_and_prompts"],
    )

    run_second_pass_cleansing_and_eog_removal(
        data_root_dir=data_root_dir,
        list_of_participants=config["participants"],
        dataset_directory_name=config["dataset_directory_name"],
        n_runs=config["number_of_runs"],
        remove_ecg=config["remove_ECG"],
        remove_veoh_and_heog=config["remove_VEOH_and_HEOG"],
        skip_ica_if_previous_runs_exist=config["skip_ica_if_previous_runs_exist"],
        supress_excessive_plots_and_prompts=config["supress_excessive_plots_and_prompts"],
    )


if __name__ == "__main__":
    import argparse

    basicConfig(format=log_message, datefmt=date_format, level=INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the appropriate dataset config .yaml file",
        default="dataset4.yaml",
    )
    args = parser.parse_args()

    main(config_filename=args.config)
