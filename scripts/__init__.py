import utils
import preprocessing
import BEM
import hexel_current_estimation
import BIDs

def main():

    # Start up
    utils.display_welcome_message_to_terminal()

    # Load parameters
    config = utils.load_config_parameters('data/dataset4_config_file.yaml')

    list_of_participants = config['list_of_participants']

    # Ensure we have all the data we need
    utils.data_integrety_checks(list_of_participants)

    # Preprocess EMEG raw data
    preprocessing.run_preprocessing(list_of_participants)

    # Save sensor level data, epoched by trial
    Create sensor-level trails
    
    # Create Bondary Element Models
    Create BEMS

    # Create the hexel current reconstructions, epoched by trial
    Do hexel current reconstructions

    # Average the hexel current reconstructions into a single participant
    Do hexel current reconstructions

    # Export data ready for BIDS format  
    ready for BIDS output

    # Run Kymata
    XYZ

    # End code with cleanup
    utils.run_cleanup()


if __name__ == "__main__":
    main()