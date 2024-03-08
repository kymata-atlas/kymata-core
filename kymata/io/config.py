from pathlib import Path

import yaml

from kymata.datasets.data_root import data_root_path
from kymata.io.file import path_type, file_type, open_or_use


def load_config(config_location: path_type | file_type):
    """Load config parameters"""
    with open_or_use(config_location) as stream:
        return yaml.safe_load(stream)


def modify_param_config(config_location: str, key: str, value):
    with open(config_location, 'r') as stream:
        """Load config parameter"""
        data = yaml.safe_load(stream)
        data[key] = value
    yaml.emitter.Emitter.process_tag = lambda self, *args, **kw: None
    with open(config_location, 'w') as file:
        yaml.dump(data,
                  file,
                  sort_keys = False)


def get_root_dir(config: dict) -> str:
    if config['data_location'] == "local":
        return str(Path(data_root_path(), "emeg_study_data")) + "/"
    elif config['data_location'] == "cbu":
        return '/imaging/projects/cbu/kymata/data/'
    elif config['data_location'] == "cbu-local":
        return '//cbsu/data/imaging/projects/cbu/kymata/data/'
    else:
        raise ValueError(
            "The `data_location` parameter in the config file must be either 'cbu' or 'local' or 'cbu-local'.")
