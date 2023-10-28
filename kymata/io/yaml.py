import yaml

def load_config_parameters(file_location: String):
    '''Load config parameters'''
    with open(file_location, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)