import yaml


def load_config_parameters(file_location: str):
    '''Load config parameters'''
    with open(file_location, "r") as stream:
        return yaml.safe_load(stream)
