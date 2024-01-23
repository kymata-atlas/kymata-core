from pathlib import Path
from colorama import Fore

from kymata.io.yaml import load_config
from kymata.io.cli import print_with_color
from kymata.preproc.data_cleansing import estimate_noise_cov


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

    estimate_noise_cov( data_root_dir = data_root_dir,
                       emeg_machine_used_to_record_data = config['emeg_machine_used_to_record_data'],
                       list_of_participants = config['list_of_participants'],
                       dataset_directory_name  = config['dataset_directory_name'],
                       n_runs = config['number_of_runs'],
                       cov_method = config['cov_method'],
                       duration_emp = config['duration'],
                        )

if __name__ == '__main__':
    main()
