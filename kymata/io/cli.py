from sys import stdout
from colorama import Style


def print_with_color(message: str, fore_color, style=Style.BRIGHT):
    """Print a message in a bright style"""
    print(f"{fore_color}{style}{message}{Style.RESET_ALL}")


def input_with_color(message: str, fore_color, style=Style.BRIGHT) -> str:
    """Get input in a bright style"""
    return input(f"{fore_color}{style}{message}{Style.RESET_ALL}")


def print_progress(iteration: int, total: int,
                   prefix: str = '',
                   suffix: str = '',
                   *,
                   decimals: int = 1,
                   bar_length: int = 100,
                   update_downsample: int = 1,
                   full_char='▒',
                   empty_char='┄',
                   terminal_char='║',
                   clear_on_completion: bool = False):
    """
    Call in a loop to create terminal progress bar.
    Based on https://github.com/emcoglab/ldm-core/blob/main/utils/log.py
    @params:
        iteration           - Required  : current iteration (Int)  0-indexed
        total               - Required  : total iterations (Int)  total n-iterations
        prefix              - Optional  : prefix string (Str)
        suffix              - Optional  : suffix string (Str)
        decimals            - Optional  : positive number of decimals in percent complete (Int)
        bar_length          - Optional  : character length of bar (Int)
        clear_on_completion - Optional  : clear the bar when it reaches 100% (bool)
    """

    iteration += 1  # 1-index

    if iteration < total and iteration % update_downsample != 0:
        stdout.flush()
        return

    portion_complete = iteration / float(total)
    percents = f"{100 * portion_complete:.{decimals}f}%"
    filled_length = int(round(bar_length * portion_complete))
    bar = (full_char * filled_length) + (empty_char * (bar_length - filled_length))

    stdout.write(f'\r{prefix}{terminal_char}{bar}{terminal_char} {percents}{suffix}'),

    if iteration == total:
        stdout.write("\r" if clear_on_completion else "\n")

    stdout.flush()
