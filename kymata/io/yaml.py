import yaml


def load_config(config_location: str):
    """Load config parameters"""
    with open(config_location, "r") as stream:
        return yaml.safe_load(stream)
