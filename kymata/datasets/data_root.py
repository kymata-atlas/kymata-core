from os import getcwd, getenv
from pathlib import Path
from typing import Optional

from kymata.io.file import PathType


_DATA_PATH_ENVIRONMENT_VAR_NAME = "KYMATA_DATA_ROOT"
DATA_DIR_NAME = "kymata-core-data"

# Places downloaded datasets could go, in order of preference
_preferred_default_data_locations = [
    Path(Path(__file__).parent.parent.parent),  # kymata/../data_dir (next to kymata dir)
    Path(getcwd()),                             # <cwd>/data_dir
    Path(Path.home(), "Documents"),             # ~/Documents/data_dir
    Path(Path.home()),                          # ~/data_dir
]


def data_root_path(data_root: Optional[PathType] = None) -> Path:

    # Check if the data root has been specified

    # Might be in an environmental variable
    if data_root is None:
        data_root: PathType | None = getenv(_DATA_PATH_ENVIRONMENT_VAR_NAME, default=None)

    # Might have been supplied as an argument
    if data_root is not None:
        if isinstance(data_root, str):
            data_root = Path(data_root)
        # Data root specified
        if not data_root.exists():
            raise FileNotFoundError(f"data_root {str(data_root)} specified but does not exist")
        if not data_root.is_dir():
            raise NotADirectoryError(f"Please specify a directory ({str(data_root)} is not a directory)")

        return data_root

    else:
        # Data root not specified

        # Check if the data root already exists
        for loc in _preferred_default_data_locations:
            if (here := Path(loc, DATA_DIR_NAME)).exists():
                data_root = here
                break

        # If not, attempt to create it
        if data_root is None:
            here: Path | None = None
            for loc in _preferred_default_data_locations:
                here = Path(loc, DATA_DIR_NAME)
                try:
                    here.mkdir()
                    break
                # If it fails for sensible reasons, no sweat, we'll fall through to the next option
                except (FileNotFoundError, OSError):
                    # Parent didn't exist, not writeable, etc
                    pass
            # Did we make it?
            if here is not None and here.exists():
                data_root = here
            else:
                raise FileNotFoundError("Failed to create data root directory")

        # Data root location has been derived, rather than prespecified, so feed that back to the user to avoid a
        # different location somehow being derived next time
        print(f"Data root set at {str(data_root)}.")
        print(f"Consider setting this as environmental variable {_DATA_PATH_ENVIRONMENT_VAR_NAME} to ensure it's reused"
              f" next time.")
        print(f"Hint: $> {_DATA_PATH_ENVIRONMENT_VAR_NAME}=\"{str(data_root)}\"")
        return data_root
