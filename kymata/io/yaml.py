from pathlib import Path

import yaml
from yaml.emitter import Emitter

from kymata.io.file import PathType


def load_config(config_location: PathType):
    """Load config parameters"""
    config_location = Path(config_location)
    with config_location.open("r") as stream:
        return yaml.safe_load(stream)


def modify_param_config(config_location: PathType, key: str, value):
    config_location = Path(config_location)
    with config_location.open("r") as stream:
        data = yaml.safe_load(stream)
        data[key] = value
    Emitter.process_tag = lambda self, *args, **kw: None
    with config_location.open("w") as file:
        yaml.dump(data,
                  file,
                  sort_keys = False)
