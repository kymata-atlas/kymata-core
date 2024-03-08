from pathlib import Path

from kymata.io.config import load_config, get_root_dir
from kymata.preproc.data_cleansing import estimate_noise_cov


def main():
    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset4.yaml")))

    estimate_noise_cov(data_root_dir = get_root_dir(config),
                       emeg_machine_used_to_record_data = config['emeg_machine_used_to_record_data'],
                       list_of_participants = config['list_of_participants'],
                       dataset_directory_name  = config['dataset_directory_name'],
                       n_runs = config['number_of_runs'],
                       cov_method = config['cov_method'],
                       duration_emp = config['duration'],
                       reg_method = config['reg_method'])


if __name__ == '__main__':
    main()
