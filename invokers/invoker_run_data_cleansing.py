from pathlib import Path

from kymata.io.config import load_config, get_root_dir
from kymata.preproc.data_cleansing import run_first_pass_cleansing_and_maxwell_filtering, \
    run_second_pass_cleansing_and_EOG_removal


def main():
    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset4.yaml")))

    data_root_dir = get_root_dir(config)

    run_first_pass_cleansing_and_maxwell_filtering(
        data_root_dir = data_root_dir,
        list_of_participants=config['list_of_participants'],
        dataset_directory_name=config['dataset_directory_name'],
        n_runs=config['number_of_runs'],
        emeg_machine_used_to_record_data=config['EMEG_machine_used_to_record_data'],
        skip_maxfilter_if_previous_runs_exist=config['skip_maxfilter_if_previous_runs_exist'],
        automatic_bad_channel_detection_requested=config['automatic_bad_channel_detection_requested'],
        supress_excessive_plots_and_prompts=config['supress_excessive_plots_and_prompts'],
    )

    run_second_pass_cleansing_and_EOG_removal(
        data_root_dir=data_root_dir,
        list_of_participants=config['list_of_participants'],
        dataset_directory_name=config['dataset_directory_name'],
        n_runs=config['number_of_runs'],
        remove_ecg=config['remove_ECG'],
        remove_veoh_and_heog=config['remove_VEOH_and_HEOG'],
        skip_ica_if_previous_runs_exist=config['skip_ica_if_previous_runs_exist'],
        supress_excessive_plots_and_prompts=config['supress_excessive_plots_and_prompts'],
    )


if __name__ == '__main__':
    main()
