from colorama import Fore
from colorama import Style
import yaml

def display_welcome_message_to_terminal():
    '''Runs welcome message'''

    print(f"{Fore.BLUE}{Style.BRIGHT}-----------------------------------------------{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT} Kymata Preprocessing and Analysis Pipeline    {Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}-----------------------------------------------{Style.RESET_ALL}")
    print("")

def run_cleanup():
    '''Runs clean up'''

    print(f"{Fore.GREEN}{Style.BRIGHT}Exited sucessfully.{Style.RESET_ALL}")

def load_config_parameters(file_location):
    '''Loads config file'''

    with open(file_location, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)



def data_integrety_checks():
    '''Runs data integrety checks before starting pipeline.'''

    Check raw files are there?

    Check create x4 bad channel .txt files?

    CHeckincluding empty room data for the covarience matrix.

    CHeck T1 and FLASH XXX
