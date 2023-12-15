from colorama import Style


def print_with_color(message: str, fore_color, style=Style.BRIGHT):
    """Print a message in a bright style"""
    print(f"{fore_color}{style}{message}{Style.RESET_ALL}")


def input_with_color(message: str, fore_color, style=Style.BRIGHT) -> str:
    """Get input in a bright style"""
    return input(f"{fore_color}{style}{message}{Style.RESET_ALL}")
