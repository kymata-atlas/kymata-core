from pathlib import Path
from colorama import Fore

from kymata.io.yaml import load_config
from kymata.io.cli import print_with_color
from kymata.preproc.hexel_current_estimation import create_forward_model_and_inverse_solution, create_hexel_morph_maps


# noinspection DuplicatedCode
def main():
    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset4.yaml")))

    if config['data_location'] == "local":
        data_root_dir = str(Path(Path(__file__).parent.parent, "kymata-toolbox-data", "emeg_study_data")) + "/"
    elif config['data_location'] == "cbu":
        data_root_dir = '/imaging/projects/cbu/kymata/data/'
    elif config['data_location'] == "cbu-local":
        data_root_dir = '//cbsu/data/imaging/projects/cbu/kymata/data/'
    else:
        raise Exception("The 'data_location' parameter in the config file must be either 'cbu' or 'local' or 'cbu-local'.")

    # create_current_estimation_prerequisites(data_root_dir, config=config)

    create_forward_model_and_inverse_solution(data_root_dir, config=config)

    create_hexel_morph_maps(data_root_dir, config=config)


def _display_welcome_message_to_terminal():
    """Runs welcome message"""
    print_with_color("-----------------------------------------------", Fore.BLUE)
    print_with_color(" Kymata Preprocessing and Analysis Pipeline    ", Fore.BLUE)
    print_with_color("-----------------------------------------------", Fore.BLUE)
    print_with_color("", Fore.BLUE)

def _run_cleanup():
    """Runs clean up"""
    print_with_color("Exited successfully.", Fore.GREEN)


if __name__ == '__main__':
    main()
