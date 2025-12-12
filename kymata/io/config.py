from pathlib import Path

import yaml

from kymata.datasets.data_root import data_root_path
from kymata.io.file import PathType, FileType, open_or_use


def load_config(config_location: PathType | FileType) -> dict:
    """
    Load configuration parameters from a specified path or file.

    This function reads the configuration parameters from a YAML file located at the given path or open file.

    Args:
        config_location (PathType | FileType): The path to the configuration file or an open file object.

    Returns:
        dict: The configuration parameters loaded from the file.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        yaml.YAMLError: If there is an error in parsing the YAML file.
    """
    with open_or_use(config_location) as stream:
        return yaml.safe_load(stream)


def modify_param_config(config_location: str, key: str, value):
    """
    Modify a specific configuration parameter in the given configuration file.

    This function updates the value of a specified key in the configuration file and saves the changes.

    Args:
        config_location (str): The path to the configuration file.
        key (str): The key of the configuration parameter to be modified.
        value: The new value to be assigned to the specified key.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If there is an error in parsing the YAML file.
    """
    with open(config_location, "r") as stream:
        """Load config parameter"""
        data = yaml.safe_load(stream)
        data[key] = value
    yaml.emitter.Emitter.process_tag = lambda self, *args, **kw: None
    with open(config_location, "w") as file:
        yaml.dump(data, file, sort_keys=False)


def get_root_dir(config: dict) -> str:
    """
    Get the root directory based on the configuration parameters.

    This function returns the appropriate root directory path based on the 'data_location' parameter
    in the provided configuration dictionary.

    Args:
        config (dict): The configuration dictionary containing the 'data_location' parameter.

    Returns:
        str: The root directory path corresponding to the 'data_location' parameter.

    Raises:
        ValueError: If the 'data_location' parameter is not 'local', 'cbu', or 'cbu-local'.
    """
    if config["data_location"] == "local":
        return str(Path(data_root_path(), "emeg_study_data")) + "/"
    elif config["data_location"] == "cbu":
        return "/imaging/projects/cbu/kymata/data/"
    elif config["data_location"] == "cbu-local":
        return "//cbsu/data/imaging/projects/cbu/kymata/data/"
    else:
        raise ValueError(
            "The `data_location` parameter in the config file must be either 'cbu' or 'local' or 'cbu-local'."
        )
