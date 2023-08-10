from colorama import Fore
from colorama import Style
import yaml

def load_config_parameters(file_location: String):
    '''Load config parameters'''
    with open(file_location, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def load_hexel_expression_master(hexel_expression_master_filename: String | none):
    '''Load lexel expression master file'''
    if filename_location is not none:
        load hexel_expression_master
        print the functions it already contains
        return hexel_expression_master
    return none