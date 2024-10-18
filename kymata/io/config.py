from pathlib import Path

import yaml

from invokers.run_gridsearch import _logger
from kymata.datasets.data_root import data_root_path
from kymata.io.file import PathType, FileType, open_or_use


def load_config(config_location: PathType | FileType):
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


def get_config_value_with_fallback(config: dict, config_key: str, fallback):
    """
    Retrieve a config value by key, with a fallback option if the key does not exist.

    This function attempts to fetch the value associated with `config_key` from the
    provided `config` dictionary. If the key is not present, it logs an error message
    and returns the specified `fallback` value.

    Args:
        config (dict): A dictionary containing configuration settings.
        config_key (str): The key for the desired configuration value.
        fallback (any): The value to return if the configuration key is not found.

    Returns:
        any: The value associated with `config_key` if it exists; otherwise, the `fallback` value.

    Raises:
        None: This function does not raise exceptions; it handles missing keys gracefully.
    """
    try:
        return config[config_key]
    except KeyError:
        _logger.error(
            f'Config did not contain any value for "{config_key}", falling back to default value {fallback}'
        )
        return fallback
