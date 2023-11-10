from dataclasses import dataclass
from os import getenv, getcwd, remove, rmdir
from pathlib import Path
from typing import Optional
from urllib import request

from kymata.io.file import path_type


_DATA_PATH_ENVIRONMENT_VAR_NAME = "KYMATA_DATA_ROOT"
_DATA_DIR_NAME = "kymata_data"

# Places downloaded datasets could go, in order of preference
_preferred_default_data_locations = [
    Path(Path(__file__).parent.parent.parent),  # kymata/../data_dir (next to kymata dir)
    Path(getcwd()),                             # <cwd>/data_dir
    Path(Path.home(), "Documents"),             # ~/Documents/data_dir
    Path(Path.home()),                          # ~/data_dir
]


@dataclass
class SampleDataset:
    """
    Info required to retrieve a dataset stored locally.

    Names in `.filenames` refer to local files, which (if `remote_root` is specified) are paired with identically
    named remote files.
    """
    name: str
    path: Path
    filenames: list[str]
    remote_root: Optional[str] = None

    def download(self):
        _download_dataset(self)


def data_root_path(data_root: Optional[path_type] = None) -> Path:

    # Check if the data root has been specified

    # Might be in an environmental variable
    if data_root is None:
        data_root: path_type | None = getenv(_DATA_PATH_ENVIRONMENT_VAR_NAME, default=None)

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
            if (here := Path(loc, _DATA_DIR_NAME)).exists():
                data_root = here
                break

        # If not, attempt to create it
        if data_root is None:
            here: Path | None = None
            for loc in _preferred_default_data_locations:
                here = Path(loc, _DATA_DIR_NAME)
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


def _download_dataset(local_dataset):
    print(f"Downloading dataset: {local_dataset.name}")
    if local_dataset.remote_root is None:
        raise ValueError("No remote root provided")
    local_dataset.path.mkdir(exist_ok=True)
    for filename in local_dataset.filenames:
        remote = local_dataset.remote_root + "/" + filename
        local = Path(local_dataset.path, filename)
        if local.exists():
            print(f"Local file already exists: {local}")
        else:
            print(f"Downloading {remote} to {local}")
            request.urlretrieve(remote, local)


def get_kymata_mirror_q3_2023(download: bool = True, data_root: Optional[path_type] = None) -> SampleDataset:
    name = "kymata_mirror_Q3_2023"

    local_dataset = SampleDataset(
        name=name,
        path=Path(data_root_path(data_root=data_root), name),
        filenames=["kymata_mirror_Q3_2023_expression_endtable.nkg"],
        remote_root="https://kymata.org/assets_kymata_toolbox_tutorial_data/gridsearch-result-data/"
    )
    if download:
        local_dataset.download()
    return local_dataset


def delete_dataset(local_dataset: SampleDataset):
    # Make sure it's not silent
    print(f"Deleting dataset {local_dataset.name}")
    # Only allow deletion if the specified url is within the data dir
    assert data_root_path() in local_dataset.path.parents, f"Cannot delete dataset outside of data root directory"
    if not local_dataset.path.exists():
        # Nothing to delete
        print(f"{str(local_dataset.path)} doesn't exist")
        return

    for file in local_dataset.filenames:
        to_delete = Path(local_dataset.path, file)
        print(f"Deleting file {str(to_delete)}")
        remove(to_delete)
    print(f"Deleting directory {str(local_dataset.path)}")
    rmdir(local_dataset.path)
