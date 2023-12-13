from kymata.preproc.utils import data_integrety_checks, display_welcome_message_to_terminal, load_config_parameters, run_cleanup
from kymata.preproc.hexel_current_estimation import create_forward_model_and_inverse_solution, create_hexel_current_files


def main():
    '''The pipeline invoker'''

    # Start up
    display_welcome_message_to_terminal()

    # Load parameters
    config = load_config_parameters('kymata/config/dataset4_config_file.yaml')

    # Ensure we have all the data we need
    data_integrety_checks(config=config)

    # Preprocess EMEG raw data
    #preprocessing.run_preprocessing(config=config)

    # Save sensor level data, epoched by trial
    #preprocessing.create_trials(config=config)
    
    # Create Boundary Element Models
    #hexel_current_estimation.create_current_estimation_prerequisites(config=config)

    # Create forward model and inverse solution
    create_forward_model_and_inverse_solution(config=config)

    # Create the hexel current reconstructions, epoched by trial
    create_hexel_current_files(config=config)

    # Average the hexel current reconstructions into a single participant
#    average_participants_hexel_currents(list_of_participants=list_of_participants, input_stream=input_stream)

    # Export data ready for BIDS format  
#    export_for_sharing()

    # Run Kymata
    #XYZ

    # End code with cleanup
    run_cleanup()


if __name__ == "__main__":
    main()