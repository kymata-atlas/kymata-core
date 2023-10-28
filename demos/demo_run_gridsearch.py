from colorama import Fore
from colorama import Style

from kymata.plot.plotting import expression_plot
from kymata.io.yml import load_config_parameters
from kymata.gridsearch.plain_gridsearch import do_gridsearch_on_both_hemsipheres

def main():

    # Start up
    print(f"{Fore.BLUE}{Style.BRIGHT}----------------------------------{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT} Kymata Toolbox                   {Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}----------------------------------{Style.RESET_ALL}")

    # Load parameters
    
    config = load_config_parameters('data/sample-data/sample_config_file.yaml')
    
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

    #predicted_function_outputs_data = load_data.load_predicted_function_outputs(neurophysiology_data_file_direcotry = neurophysiology_data_file_direcotry)

    # Run search for function output
    #expressionSet = do_gridsearch_on_both_hemsipheres(neurophysiology_data_file_directory,
    #                                            predicted_function_outputs_data,
    #                                            functions_to_apply_gridsearch)
    
    expression_plot(expressionSet, include_functions=["hornschunck_horizontalVelocity"])
