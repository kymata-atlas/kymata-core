from abc import ABC, abstractmethod
from os import getenv, getcwd, remove, rmdir
from pathlib import Path
from typing import Optional
from urllib import request

from kymata.entities.expression import HexelExpressionSet, SensorExpressionSet
from kymata.io.file import path_type
from kymata.io.nkg import load_expression_set

_DATA_PATH_ENVIRONMENT_VAR_NAME = "KYMATA_DATA_ROOT"
_DATA_DIR_NAME = "kymata-toolbox-data"

# Places downloaded datasets could go, in order of preference
_preferred_default_data_locations = [
    Path(Path(__file__).parent.parent.parent),  # kymata/../data_dir (next to kymata dir)
    Path(getcwd()),                             # <cwd>/data_dir
    Path(Path.home(), "Documents"),             # ~/Documents/data_dir
    Path(Path.home()),                          # ~/data_dir
]


class SampleDataset(ABC):
    """
    Info required to retrieve a dataset stored locally.

    Names in `.filenames` refer to local files, which (if `remote_root` is specified) are paired with identically
    named remote files.
    """

    def __init__(self,
                 name: str,
                 filenames: list[str],
                 data_root: Optional[path_type],
                 remote_root: Optional[str],
                 download: bool):
        self.name: str = name
        self.filenames: list[str] = filenames
        self.data_root: Path = data_root_path(data_root)
        self.remote_root: str = remote_root

        if download:
            self.download()

    @property
    def path(self) -> Path:
        return Path(self.data_root, self.name)

    def download(self):
        print(f"Downloading dataset: {self.name}")
        if self.remote_root is None:
            raise ValueError("No remote root provided")
        self.path.mkdir(exist_ok=True)
        for filename in self.filenames:
            remote = self.remote_root + "/" + filename
            local = Path(self.path, filename)
            if local.exists():
                print(f"Local file already exists: {local}")
            else:
                print(f"Downloading {remote} to {local}")
                request.urlretrieve(remote, local)

    @abstractmethod
    def to_expressionset(self) -> HexelExpressionSet:
        raise NotImplementedError()


class KymataMirror2023Q3Dataset(SampleDataset):
    def __init__(self, data_root: Optional[path_type] = None, download: bool = True):
        name = "kymata_mirror_Q3_2023"
        super().__init__(
            name=name,
            filenames=[
                "kymata_mirror_Q3_2023_expression_endtable.nkg",
            ],
            data_root=data_root,
            remote_root="https://kymata.org/assets_kymata_toolbox_tutorial_data/gridsearch-result-data/",
            download=download,
        )

    def to_expressionset(self) -> HexelExpressionSet:
        es = load_expression_set(from_path_or_file=Path(self.path, self.filenames[0]))
        assert isinstance(es, HexelExpressionSet)
        return es


class TVLInsLoudnessOnlyDataset(SampleDataset):
    def __init__(self, data_root: Optional[path_type] = None, download: bool = True):
        name = "TVL_2020_ins_loudness_only"
        super().__init__(
            name=name,
            filenames=[
                "TVL_2020_ins_loudness_only.nkg",
            ],
            data_root=data_root,
            remote_root="https://kymata.org/assets_kymata_toolbox_tutorial_data/gridsearch-result-data/",
            download=download,
        )

    def to_expressionset(self) -> HexelExpressionSet:
        es = load_expression_set(from_path_or_file=Path(self.path, self.filenames[0]))
        assert isinstance(es, HexelExpressionSet)
        return es


class TVLDeltaInsTC1LoudnessOnlyDataset(SampleDataset):
    def __init__(self, data_root: Optional[path_type] = None, download: bool = True):
        name = "TVL_2020_delta_ins_tontop_chan1_loudness_only"
        super().__init__(
            name=name,
            filenames=[
                "TVL_2020_delta_ins_tontop_chan1_loudness_only.nkg"
            ],
            data_root=data_root,
            remote_root="https://kymata.org/assets_kymata_toolbox_tutorial_data/gridsearch-result-data/",
            download=download,
        )

    def to_expressionset(self) -> HexelExpressionSet:
        es = load_expression_set(from_path_or_file=Path(self.path, self.filenames[0]))
        assert isinstance(es, HexelExpressionSet)
        return es

class TVLDeltaInsTC1LoudnessOnlySensorsDataset(SampleDataset):
    def __init__(self, data_root: Optional[path_type] = None, download: bool = True):
        name = "TVL_2020_delta_ins_tontop_chan1_loudness_only_sensors"
        super().__init__(
            name=name,
            filenames=[
                "TVL_2020_delta_ins_tontop_chan1_loudness_only_sensors.nkg"
            ],
            data_root=data_root,
            remote_root="https://kymata.org/assets_kymata_toolbox_tutorial_data/gridsearch-result-data/",
            download=download,
        )

    def to_expressionset(self) -> SensorExpressionSet:
        es = load_expression_set(from_path_or_file=Path(self.path, self.filenames[0]))
        assert isinstance(es, SensorExpressionSet)
        return es


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
