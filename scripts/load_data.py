from colorama import Fore
from colorama import Style
import yaml


#    run_number = load_data.load_predicted_function_outputs(neurophysiology_data_file_direcotry = neurophysiology_data_file_direcotry)
#    check if function_name or link already exists
#check function names etc aren't blank


def load_config_parameters(file_location):
    with open(file_location, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)