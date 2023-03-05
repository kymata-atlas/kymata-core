from colorama import Fore
from colorama import Style

def display_welcome_message_to_terminal():
    print(f"{Fore.BLUE}{Style.BRIGHT}----------------------------------{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT} Kymata Toolbox                   {Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}----------------------------------{Style.RESET_ALL}")
    print("")

def run_cleanup():
    '''Runs clean up'''

    print(f"{Fore.GREEN}{Style.BRIGHT}Exited sucessfully.{Style.RESET_ALL}")
