import yaml


def load_config_parameters(file_location: str):
    """Loads config file"""

    with open(file_location, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def load_recording_config(file_location: str):
    """Loads EMEG bad channels from yaml suitable YAML file"""

    with open(file_location, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def data_integrety_checks(config: dict):
    """Runs data integrety checks before starting pipeline."""

    #Check raw files are there?

    #Check create x4 bad channel .txt files?

    #CHeckincluding empty room data for the covarience matrix.

    #CHeck T1 and FLASH XXX
