from colorama import Fore, Style


def display_welcome_message_to_terminal():
    """Runs welcome message"""

    print(f"{Fore.BLUE}{Style.BRIGHT}-----------------------------------------------{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT} Kymata Preprocessing and Analysis Pipeline    {Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}-----------------------------------------------{Style.RESET_ALL}")
    print("")


def run_cleanup():
    """Runs clean up"""

    print(f"{Fore.GREEN}{Style.BRIGHT}Exited sucessfully.{Style.RESET_ALL}")
