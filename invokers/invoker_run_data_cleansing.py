from pathlib import Path
from colorama import Fore

from kymata.io.yaml import load_config
from kymata.io.cli import print_with_color
from kymata.preproc.data_cleansing import run_first_pass_cleansing_and_maxwell_filtering, run_second_pass_cleansing_and_EOG_removal


# noinspection DuplicatedCode
def main():
    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset4.yaml")))

    if config['data_location'] == "local":
        data_root_dir = str(Path(Path(__file__).parent.parent, "kymata-toolbox-data", "emeg_study_data")) + "/"
    elif config['data_location'] == "cbu":
        data_root_dir = '/imaging/projects/cbu/kymata/data/'
    elif config['data_location'] == "cbu-local":
        data_root_dir = '//cbsu/data/imaging/projects/cbu/kymata/data/'
    else:
        raise Exception("The 'data_location' parameter in the config file must be either 'cbu' or 'local'.")

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

def _display_welcome_message_to_terminal():
    """Runs welcome message"""
    print_with_color("-----------------------------------------------", Fore.BLUE)
    print_with_color(" Kymata Preprocessing and Analysis Pipeline    ", Fore.BLUE)
    print_with_color("-----------------------------------------------", Fore.BLUE)
    print_with_color("", Fore.BLUE)

def _run_cleanup():
    """Runs clean up"""
    print_with_color("Exited successfully.", Fore.GREEN)

if __name__ == '__main__':
    main()
