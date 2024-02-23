import yaml


def load_config(config_location: str):
    """Load config parameters"""
    with open(config_location, "r") as stream:
        return yaml.safe_load(stream)


def modify_param_config(config_location: str, key: str, value):
    with open(config_location, 'r') as stream:
        """Load dataset_config parameter"""
        data = yaml.safe_load(stream)
        data[key] = value
    yaml.emitter.Emitter.process_tag = lambda self, *args, **kw: None
    with open(config_location, 'w') as file:
        yaml.dump(data,
                  file,
                  sort_keys = False)