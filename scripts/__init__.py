import utils
import preprocessing
#import bem
#import hexel_current_estimation
#import bids

def main():

    '''The pipeline invoker'''

    # Start up
    utils.display_welcome_message_to_terminal()

    # Load parameters
    config = utils.load_config_parameters('data/configs/dataset4_config_file.yaml')

    list_of_participants = config['list_of_participants']
    input_stream = config['input_stream']
    remove_ECG = config['remove_ECG']
    remove_VEOH_and_HEOG = config['remove_VEOH_and_HEOG']

    # Ensure we have all the data we need
    utils.data_integrety_checks(list_of_participants=list_of_participants)

    # Preprocess EMEG raw data
    preprocessing.run_preprocessing(list_of_participants=list_of_participants,
                                    input_stream=input_stream,
                                    remove_ECG=remove_ECG,
                                    remove_VEOH_and_HEOG=remove_VEOH_and_HEOG)

    # Save sensor level data, epoched by trial
#    preprocessing.create_trials(list_of_participants = list_of_participants, input_stream = input_stream)
    
    # Create Boundary Element Models
#    bem.create_boundary_element_model(list_of_participants = list_of_participants, input_stream = input_stream)

    # Create forward model and inverse solution
#    hexel_current_estimation.create_forward_model_and_inverse_solution(list_of_participants = list_of_participants, input_stream = input_stream)

    # Create the hexel current reconstructions, epoched by trial
#    hexel_current_estimation.create_hexel_current_files(list_of_participants=list_of_participants, input_stream=input_stream)

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