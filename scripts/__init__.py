import utils
import load_data
import gridsearch
import plotting

def main():

    # Start up
    utils.display_welcome_message_to_terminal()

    # Load parameters
    
    #config = load_data.load_config_parameters('data/dataset4_config_file.yaml')
    
    #neurophysiology_data_file_directory = config['neurophysiology_data_file_directory']
    #hexel_expression_master_filename = config['neurophysiology_data_file_direcotry']
    #is_sensor_esimation = config['is_sensor_esimation']
    #is_debug_mode = config['is_debug_mode']
    #functions_to_apply_gridsearch = config['functions_to_apply_gridsearch']
    #functions_to_be_plotted = config['functions_to_be_plotted']
    #force_overwrite_hexel_expression_master = config['force_overwrite_hexel_expression_master ']
    #overwrite_hexel_expression_master_file = config['force_overwrite_hexel_expression_master ']

    #print(paramenters for viewer)
    
    # Load data

    #neurophysiology_data = load_data.load_neurophysiology_data(neurophysiology_data_file_direcotry = neurophysiology_data_file_direcotry)
    #predicted_function_outputs_data = load_data.load_predicted_function_outputs(neurophysiology_data_file_direcotry = neurophysiology_data_file_direcotry)

    # Run search for function output
    
    #hexel_expression = gridsearch.xxxdo(neurophysiology_data, predicted_function_outputs_data, hexel_expression, functions_to_apply_gridsearch)
    #hexel_expression merge flipped together

    # save hexel
    
    # hexel_expression = utils.xxxdo()
    # save for kymata atlas encoded = utils.xxx()

    # Plot expression plots
    #    [2]plotting.plot_expression_plot(is_sensor_esimation = is_sensor_esimation, hexel_expression = hexel_expression, (functions_of_interests = functions_of_interest))
    plotting.plot_expression_plot()

    # End code with cleanup
    utils.run_cleanup()


if __name__ == "__main__":
    main()