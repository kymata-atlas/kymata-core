import utils
import preprocessing
import hexel_current_estimation
#import bids

def main():

    '''The pipeline invoker'''

    # Start up
    utils.display_welcome_message_to_terminal()

    # Load parameters
    config = utils.load_config_parameters('data/configs/dataset4_config_file.yaml')

    # Ensure we have all the data we need
    utils.data_integrety_checks(config=config)

    # Preprocess EMEG raw data
    #preprocessing.run_preprocessing(config=config)

    # Save sensor level data, epoched by trial
    #preprocessing.create_trials(config=config)
    
    # Create Boundary Element Models
    #hexel_current_estimation.create_current_estimation_prerequisites(config=config)

    # Create forward model and inverse solution
    hexel_current_estimation.create_forward_model_and_inverse_solution(config=config)

    # Create the hexel current reconstructions, epoched by trial
    hexel_current_estimation.create_hexel_current_files(config=config)

    # Average the hexel current reconstructions into a single participant
#    hexel_current_estimation.average_participants_hexel_currents(list_of_participants=list_of_participants, input_stream=input_stream)

    # Export data ready for BIDS format  
#    bids.export_for_sharing()

    # Run Kymata
    #XYZ

    # End code with cleanup
    utils.run_cleanup()


if __name__ == "__main__":
    main()