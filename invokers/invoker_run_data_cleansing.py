from pathlib import Path
from colorama import Fore

from kymata.io.yaml import load_config
from kymata.io.cli import print_with_color
from kymata.preproc.data_cleansing import run_preprocessing, create_trials


# noinspection DuplicatedCode
def main():
    config = load_config(str(Path(Path(__file__).parent, "kymata", "config", "dataset4.yaml")))

    run_first_pass_cleansing_and_maxwell_filtering(
        list_of_participants=config['list_of_participants'],
        dataset_directory_name=config['dataset_directory_name'],
        n_runs=config['number_of_runs'],
        emeg_machine_used_to_record_data=config['EMEG_machine_used_to_record_data'],
        skip_maxfilter_if_previous_runs_exist=config['skip_maxfilter_if_previous_runs_exist'],
        automatic_bad_channel_detection_requested=config['automatic_bad_channel_detection_requested'],
    )

    run_second_pass_cleansing_and_EOG_removal(
        list_of_participants=config['list_of_participants'],
        dataset_directory_name=config['dataset_directory_name'],
        n_runs=config['number_of_runs'],
        remove_ecg=config['remove_ECG'],
        remove_veoh_and_heog=config['remove_VEOH_and_HEOG'],
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
