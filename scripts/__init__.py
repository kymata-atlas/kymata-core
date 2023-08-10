import utils
import load_data
import gridsearch
import plotting

def main():

    # Start up
    utils.display_welcome_message_to_terminal()

    # Load parameters
    
    config = load_data.load_config_parameters('data/sample-data/sample_config_file.yaml')
    
    neurophysiology_data_file_directory = config['neurophysiology_data_file_directory']
#    predicted_function_outputs_data = config[]
    hexel_expression_master_filename = config['hexel_expression_master_filename']
    is_sensor_esimation = config['is_sensor_esimation']
    is_debug_mode = config['is_debug_mode']
    functions_to_apply_gridsearch = config['functions_to_apply_gridsearch']
    functions_to_use_in_model_selection = config['functions_to_use_in_model_selection']
    functions_to_be_plotted = config['functions_to_be_plotted']
    inputstream = config['inputstream']

    print(config)
    
    # Load data

    #hexel_expression_master = load_data.hexel_expression_master(hexel_expression_master_filename = hexel_expression_master_filename)
    #predicted_function_outputs_data = load_data.load_predicted_function_outputs(neurophysiology_data_file_direcotry = neurophysiology_data_file_direcotry)

    # Run search for function output
    #hexel_expression_master = gridsearch.do_gridsearch(neurophysiology_data_file_directory,
    #                                            predicted_function_outputs_data,
    #                                            hexel_expression_master,
    #                                            functions_to_apply_gridsearch)

    # save hexel_expression_master
    #utils.save_hexel_expression_master(hexel_expression_master)
    
    # hexel_expression = utils.xxxdo()
    # save for kymata atlas encoded = utils.xxx()

    # Plot expression plots
    #    [2]plotting.plot_expression_plot(is_sensor_esimation = is_sensor_esimation, hexel_expression = hexel_expression, (functions_of_interests = functions_of_interest))
    plotting.plot_expression_plot()

    # End code with cleanup
    utils.run_cleanup()


if __name__ == "__main__":
    main()