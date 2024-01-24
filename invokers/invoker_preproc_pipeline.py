from colorama import Fore

import sys
sys.path.append('/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox')

from kymata.io.cli import print_with_color
from kymata.io.yaml import load_config
from kymata.preproc.data import data_integrety_checks
from kymata.preproc.pipeline import run_preprocessing, create_trials
from kymata.preproc.hexel_current_estimation import create_forward_model_and_inverse_solution, \
    create_hexel_current_files, create_current_estimation_prerequisites


def main():
    """The pipeline invoker"""

    # Start up
    # _display_welcome_message_to_terminal()

    # Load parameters
    config = load_config('/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox/kymata/config/dataset4.yaml')

    # Ensure we have all the data we need
    data_integrety_checks(config=config)

    # Preprocess EMEG raw data
    """run_preprocessing(list_of_participants=config['list_of_participants'],
                      dataset_directory_name=config['dataset_directory_name'],
                      n_runs=config['number_of_runs'],
                      emeg_machine_used_to_record_data=config['EMEG_machine_used_to_record_data'],
                      remove_ecg=config['remove_ECG'],
                      skip_maxfilter_if_previous_runs_exist=config['skip_maxfilter_if_previous_runs_exist'],
                      remove_veoh_and_heog=config['remove_VEOH_and_HEOG'],
                      automatic_bad_channel_detection_requested=config['automatic_bad_channel_detection_requested'])"""

    # Create Boundary Element Models
    # Average the hexel current reconstructions into a single participant

    create_current_estimation_prerequisites(config=config)

    # Create forward model and inverse solution
    create_forward_model_and_inverse_solution(config=config)

    # Create the hexel current reconstructions, epoched by trial
    #create_hexel_current_files(config=config)

    # Average the hexel current reconstructions into a single participant
    #    average_participants_hexel_currents(list_of_participants=list_of_participants, input_stream=input_stream)

    # Export data ready for BIDS format  
    #    export_for_sharing()

    # Run Kymata
    #    XYZ

    # End code with cleanup
    # _run_cleanup()


def _display_welcome_message_to_terminal():
    """Runs welcome message"""
    print_with_color("-----------------------------------------------", Fore.BLUE)
    print_with_color(" Kymata Preprocessing and Analysis Pipeline    ", Fore.BLUE)
    print_with_color("-----------------------------------------------", Fore.BLUE)
    print_with_color("", Fore.BLUE)


def _run_cleanup():
    """Runs clean up"""
    print_with_color("Exited successfully.", Fore.GREEN)


if __name__ == "__main__":
    main()
