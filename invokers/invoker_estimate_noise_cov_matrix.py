from pathlib import Path

from kymata.io.config import load_config, get_root_dir
from kymata.preproc.data_cleansing import estimate_noise_cov


def main(config_filename: str):
    config = load_config(str(Path(Path(__file__).parent.parent, "dataset_config", config_filename)))

    estimate_noise_cov(data_root_dir = get_root_dir(config),
                       emeg_machine_used_to_record_data = config['emeg_machine_used_to_record_data'],
                       list_of_participants = config['participants'],
                       dataset_directory_name  = config['dataset_directory_name'],
                       n_runs = config['number_of_runs'],
                       cov_method = config['cov_method'],
                       duration_emp = config['duration'],
                       reg_method = config['reg_method'])


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Create Trialwise Data')
    parser.add_argument('--config', type=str, default="dataset4.yaml")
    args = parser.parse_args()
    main(config_filename=args.config)
