import yaml


# Type alias
Config = dict


def load_config_parameters(config_location: str):
    """Loads config file"""

    with open(config_location, "r") as stream:
        config = yaml.safe_load(stream)
        assert isinstance(config, Config)
        return config


def load_recording_config(config_location: str):
    """Loads EMEG bad channels from yaml suitable YAML file"""

    with open(config_location, "r") as stream:
        config = yaml.safe_load(stream)
        assert isinstance(config, Config)
        return config
