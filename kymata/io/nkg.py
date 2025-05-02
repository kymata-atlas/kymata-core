from enum import StrEnum

from packaging import version
from io import TextIOWrapper
from pathlib import Path
from typing import Any
from warnings import warn
from zipfile import ZipFile, ZIP_LZMA

from numpy import frombuffer, int64, float64
from numpy.typing import NDArray
from sparse import COO

from kymata.entities.datatypes import HexelDType, LatencyDType, TransformNameDType, SensorDType
from kymata.entities.expression import (
    ExpressionSet, BLOCK_LEFT, BLOCK_RIGHT, BLOCK_SCALP, HexelExpressionSet, SensorExpressionSet)
from kymata.io.layouts import SensorLayout, MEGLayout, EEGLayout
from kymata.math.probability import p_to_logp
from kymata.entities.sparse_data import expand_dims
from kymata.io.file import PathType, FileType, open_or_use


class _Keys(StrEnum):
    channels = "channels"
    latencies = "latencies"
    transforms = "functions"
    data = "data"
    meg_layout = "meg-layout"
    eeg_layout = "eeg-layout"

    expressionset_type = "expressionset-type"


class _ExpressionSetTypeIdentifier(StrEnum):
    hexel = "hexel"
    sensor = "sensor"

    @classmethod
    def from_expression_set(cls, expression_set: ExpressionSet):
        if isinstance(expression_set, HexelExpressionSet):
            return cls.hexel
        elif isinstance(expression_set, SensorExpressionSet):
            return cls.sensor
        else:
            raise NotImplementedError()

    def block_names(self) -> list[str]:
        if self == self.hexel:
            return [BLOCK_LEFT, BLOCK_RIGHT]
        elif self == self.sensor:
            return [BLOCK_SCALP]
        else:
            raise NotImplementedError()


# Format versioning of the saved file. Used to (eventually) ensure old saved files can always be loaded.
#
# - Increment MINOR versions when creating a breaking change, but only when we commit to indefinitely supporting
#   the new version.
# - Increment MAJOR version when creating a breaking change which makes the format fundamentally incompatible
#   with previous versions.
#
# All distinct versions should be documented. See `nkg_compatibility.py` for changes.
#
# This value should be saved into file called /_metadata/format-version.txt within an archive.
CURRENT_VERSION = "0.5"


def file_version(from_path_or_file: PathType | FileType) -> version.Version:
    with open_or_use(from_path_or_file, mode="rb") as archive, ZipFile(
        archive, "r"
    ) as zf:
        with TextIOWrapper(
            zf.open("_metadata/format-version.txt"), encoding="utf-8"
        ) as f:
            return version.parse(str(f.read()).strip())


def load_expression_set(from_path_or_file: PathType | FileType | list[PathType]) -> ExpressionSet:
    """
    Loads an ExpressionSet from the specified path(s) or open file.

    The function determines the type of ExpressionSet (HexelExpressionSet or SensorExpressionSet)
    based on the data loaded from the provided path or file. It then constructs and returns an
    instance of the appropriate ExpressionSet subclass.

    Args:
        from_path_or_file (PathType | FileType | list[PathType]): The path, file, or list of paths from which to load
                                                                      the data.

    Returns:
        ExpressionSet: An instance of either HexelExpressionSet or SensorExpressionSet,
                           depending on the type identifier in the data.

    Raises:
        KeyError: If required keys are missing in the data dictionary.
        ValueError: If the type identifier is not recognized.
    """

    if isinstance(from_path_or_file, list):
        if len(from_path_or_file) == 0:
            raise ValueError("Must supply at least one path")
        first_path = from_path_or_file[0]
        # Head
        first_es = load_expression_set(first_path)
        if len(from_path_or_file) == 1:
            # Supplied a list of one path
            return first_es
        # Tail
        return first_es + load_expression_set(from_path_or_file[1:])

    _v, data_dict = _load_data(from_path_or_file)

    type_identifier = data_dict[_Keys.expressionset_type]

    if type_identifier == _ExpressionSetTypeIdentifier.hexel:
        return HexelExpressionSet(
            transforms=data_dict[_Keys.transforms],
            hexels_lh=[HexelDType(c) for c in data_dict[_Keys.channels][BLOCK_LEFT]],
            hexels_rh=[HexelDType(c) for c in data_dict[_Keys.channels][BLOCK_RIGHT]],
            latencies=data_dict[_Keys.latencies],
            data_lh=data_dict[_Keys.data][BLOCK_LEFT],
            data_rh=data_dict[_Keys.data][BLOCK_RIGHT],
        )
    elif type_identifier == _ExpressionSetTypeIdentifier.sensor:
        meg_layout = data_dict[_Keys.meg_layout]
        eeg_layout = data_dict[_Keys.eeg_layout]
        if meg_layout is None and eeg_layout is None:
            # If both layouts are missing, no layouts were specified and this is fine
            sensor_layout = None
        else:
            # If either one was specified, we warn about the missing one
            if meg_layout is None:
                warn("No MEG sensor layout was specified in the NKG file.")
            if eeg_layout is None:
                warn("No EEG sensor layout was specified in the NKG file.")
            sensor_layout = SensorLayout(meg=meg_layout, eeg=eeg_layout)
        return SensorExpressionSet(
            transforms=data_dict[_Keys.transforms],
            sensors=[SensorDType(c) for c in data_dict[_Keys.channels][BLOCK_SCALP]],
            latencies=data_dict[_Keys.latencies],
            data=data_dict[_Keys.data][BLOCK_SCALP],
            sensor_layout=sensor_layout,
        )


def save_expression_set(
    expression_set: ExpressionSet,
    to_path_or_file: PathType | FileType,
    compression=ZIP_LZMA,
    overwrite: bool = False,
):
    """
    Save the given ExpressionSet to a specified path or an already open file.

    This function saves the ExpressionSet data into a compressed file format.
    If a file path is provided, it creates and writes to the file. If an open file is supplied,
    it should be opened in "wb" mode. The overwrite flag is ignored if an open file is supplied.

    Args:
        expression_set (ExpressionSet): The ExpressionSet object to be saved.
        to_path_or_file (PathType | FileType): The path or open file where the ExpressionSet will be saved.
        compression: The compression method to use (default is ZIP_LZMA).
        overwrite (bool): If True, allows overwriting an existing file (default is False).

    Raises:
        FileExistsError: If the specified path already exists and overwrite is False.
        TypeError: If the provided path or file type is invalid.

    Notes:
        - The compression parameter should be compatible with the `ZipFile` class.
        - The function writes various metadata and data blocks in a structured format within the zip file.
    """

    if isinstance(to_path_or_file, str):
        to_path_or_file = Path(to_path_or_file)
    if isinstance(to_path_or_file, Path) and to_path_or_file.exists() and not overwrite:
        raise FileExistsError(to_path_or_file)

    with open_or_use(to_path_or_file, mode="wb") as f, ZipFile(f, "w", compression=compression) as zf:
        zf.writestr("_metadata/format-version.txt", CURRENT_VERSION)
        zf.writestr("_metadata/expression-set-type.txt",
                    _ExpressionSetTypeIdentifier.from_expression_set(expression_set))
        zf.writestr("/latencies.txt", "\n".join(str(x) for x in expression_set.latencies))
        zf.writestr("/transforms.txt", "\n".join(str(x) for x in expression_set.transforms))
        zf.writestr("/blocks.txt",    "\n".join(str(x) for x in expression_set._block_names))
        # Write layout for SensorExpressionSets
        layout_txt = ""
        if isinstance(expression_set, SensorExpressionSet):
            # It is valid for no sensor layout to be specified, in which case we write nothing.
            # If the sensor layout is specified, we will write something for both sensor types
            if expression_set.sensor_layout is not None:
                layout_txt += f"MEG:\t{expression_set.sensor_layout.meg!s}\n"
                layout_txt += f"EEG:\t{expression_set.sensor_layout.eeg!s}\n"
        zf.writestr("/layout.txt", layout_txt)

        for block_name in expression_set._block_names:
            zf.writestr(f"/{block_name}/channels.txt",     "\n".join(str(x) for x in expression_set._channels[block_name]))
            zf.writestr(f"/{block_name}/coo-coords.bytes", expression_set._data[block_name].data.coords.tobytes(order="C"))
            zf.writestr(f"/{block_name}/coo-data.bytes",   expression_set._data[block_name].data.data.tobytes(order="C"))
            # The shape can be inferred, but we save it as an extra validation
            zf.writestr(f"/{block_name}/coo-shape.txt",    "\n".join(str(x) for x in expression_set._data[block_name].data.shape))


def _load_data(from_path_or_file: PathType | FileType) -> tuple[version.Version, dict[str, Any]]:
    """
    Load an ExpressionSet from an open file, or the file at the specified path.

    If an open file is supplied, it should be opened in "rb" mode.
    """

    if isinstance(from_path_or_file, str):
        from_path_or_file = Path(from_path_or_file)

    # Check file version
    v: version.Version = file_version(from_path_or_file)
    if v < version.parse(CURRENT_VERSION):
        warn("This file uses an old format. Please consider re-saving the data to avoid future incompatibility.")
    # Loading old versions
    # For each, delegate to the appropriate _load_data_x_y() function, then
    # ensure the keys are set correctly.
    if v <= version.parse("0.1"):
        from kymata.io.nkg_compatibility import _load_data_0_1

        dict_0_1 = _load_data_0_1(from_path_or_file)

        # v0.1 data was stored as p values
        data = dict()
        for block, sparse_data_dict in dict_0_1["data"].items():
            sparse_data_dict["data"] = p_to_logp(sparse_data_dict["data"])

            sparse_data = COO(
                coords=sparse_data_dict["coords"],
                data=sparse_data_dict["data"],
                shape=sparse_data_dict["shape"],
                prune=True,
                fill_value=0.0,
            )
            assert sparse_data.shape == (
                len(dict_0_1["hexels"]),
                len(dict_0_1["latencies"]),
                len(dict_0_1["functions"]),
            )
            # In case there was only 1 function and we have a 2-d data matrix
            if len(sparse_data.shape) == 2:
                sparse_data = expand_dims(sparse_data)
            data[block] = sparse_data

        return_dict = {
            _Keys.channels: {
                block_name: dict_0_1["hexels"]
                for block_name in _ExpressionSetTypeIdentifier.hexel.block_names()
            },
            _Keys.transforms: dict_0_1["functions"],
            _Keys.latencies: dict_0_1["latencies"],
            _Keys.data: data,
            # Keys not present in v0.1
            _Keys.expressionset_type: _ExpressionSetTypeIdentifier.hexel,
            _Keys.meg_layout: None,
            _Keys.eeg_layout: None,
        }
    elif v <= version.parse("0.2"):
        from kymata.io.nkg_compatibility import _load_data_0_2

        dict_0_2 = _load_data_0_2(from_path_or_file)

        # v0.2 data was stored as p-values
        data = dict()
        for block, sparse_data_dict in dict_0_2["data"].items():
            sparse_data_dict["data"] = p_to_logp(sparse_data_dict["data"])

            sparse_data = COO(
                coords=sparse_data_dict["coords"],
                data=sparse_data_dict["data"],
                shape=sparse_data_dict["shape"],
                prune=True,
                fill_value=0.0,
            )
            assert sparse_data.shape == (
                len(dict_0_2["channels"]),
                len(dict_0_2["latencies"]),
                len(dict_0_2["functions"]),
            )
            # In case there was only 1 function and we have a 2-d data matrix
            if len(sparse_data.shape) == 2:
                sparse_data = expand_dims(sparse_data)
            data[block] = sparse_data
        expressionset_type = dict_0_2["expressionset-type"]
        return_dict = {
            _Keys.expressionset_type: expressionset_type,
            _Keys.channels: {
                block_name: dict_0_2["channels"]
                for block_name in _ExpressionSetTypeIdentifier(
                    expressionset_type
                ).block_names()
            },
            _Keys.transforms: dict_0_2["functions"],
            _Keys.latencies: dict_0_2["latencies"],
            _Keys.data: data,
            # Keys not present in v0.2
            _Keys.meg_layout: None,
            _Keys.eeg_layout: None,
        }
    elif v <= version.parse("0.3"):
        from kymata.io.nkg_compatibility import _load_data_0_3

        dict_0_3 = _load_data_0_3(from_path_or_file)

        # v0.2 data was stored as p-values
        data = dict()
        for block, sparse_data_dict in dict_0_3["data"].items():
            sparse_data_dict["data"] = p_to_logp(sparse_data_dict["data"])

            sparse_data = COO(
                coords=sparse_data_dict["coords"],
                data=sparse_data_dict["data"],
                shape=sparse_data_dict["shape"],
                prune=True,
                fill_value=0.0,
            )
            assert sparse_data.shape == (
                len(dict_0_3["channels"]),
                len(dict_0_3["latencies"]),
                len(dict_0_3["functions"]),
            )
            # In case there was only 1 function and we have a 2-d data matrix
            if len(sparse_data.shape) == 2:
                sparse_data = expand_dims(sparse_data)
            data[block] = sparse_data
        expressionset_type = dict_0_3["expressionset-type"]
        return_dict = {
            _Keys.expressionset_type: expressionset_type,
            _Keys.channels: {
                block_name: dict_0_3["channels"]
                for block_name in _ExpressionSetTypeIdentifier(expressionset_type).block_names()
            },
            _Keys.transforms: dict_0_3["functions"],
            _Keys.latencies: dict_0_3["latencies"],
            _Keys.data: data,
            # Keys not present in v0.3
            _Keys.meg_layout: None,
            _Keys.eeg_layout: None,
        }
    elif v <= version.parse("0.4"):
        from kymata.io.nkg_compatibility import _load_data_0_4

        dict_0_4 = _load_data_0_4(from_path_or_file)

        data = dict()
        for block, sparse_data_dict in dict_0_4["data"].items():
            sparse_data_dict["data"] = sparse_data_dict["data"]

            sparse_data = COO(
                coords=sparse_data_dict["coords"],
                data=sparse_data_dict["data"],
                shape=sparse_data_dict["shape"],
                prune=True,
                fill_value=0.0,
            )
            assert sparse_data.shape == (
                len(dict_0_4["channels"][block]),
                len(dict_0_4["latencies"]),
                len(dict_0_4["transforms"]),
            )
            # In case there was only 1 function and we have a 2-d data matrix
            if len(sparse_data.shape) == 2:
                sparse_data = expand_dims(sparse_data)
            data[block] = sparse_data
        expressionset_type = dict_0_4["expressionset-type"]
        return_dict = {
            _Keys.expressionset_type: expressionset_type,
            _Keys.channels: {
                block_name: dict_0_4["channels"][block_name]
                for block_name in _ExpressionSetTypeIdentifier(expressionset_type).block_names()
            },
            _Keys.transforms: dict_0_4["transforms"],
            _Keys.latencies: dict_0_4["latencies"],
            _Keys.data: data,
            # Keys not present in v0.4
            _Keys.meg_layout: None,
            _Keys.eeg_layout: None,
        }
    else:
        return_dict = _load_data_current(from_path_or_file)

    return v, return_dict


# noinspection DuplicatedCode
def _load_data_current(from_path_or_file: PathType | FileType) -> dict[str, Any]:
    """
    Load data from current version
    """
    return_dict = dict()
    with open_or_use(from_path_or_file, mode="rb") as archive, ZipFile(archive, "r") as zf:
        with TextIOWrapper(zf.open("_metadata/expression-set-type.txt"), encoding="utf-8") as f:
            return_dict[_Keys.expressionset_type] = str(f.read()).strip()
            expression_set_type = _ExpressionSetTypeIdentifier(return_dict[_Keys.expressionset_type])
        with TextIOWrapper(zf.open("/blocks.txt"), encoding="utf-8") as f:
            blocks = [
                str(line.strip())
                for line in f.readlines()
            ]
        with TextIOWrapper(zf.open("/latencies.txt"), encoding="utf-8") as f:
            return_dict[_Keys.latencies] = [
                LatencyDType(lat.strip())
                for lat in f.readlines()
            ]
        with TextIOWrapper(zf.open("/transforms.txt"), encoding="utf-8") as f:
            return_dict[_Keys.transforms] = [
                TransformNameDType(fun.strip())
                for fun in f.readlines()
            ]
        # If the type is sensor we will need a layout
        with TextIOWrapper(zf.open("/layout.txt"), encoding="utf-8") as f:
            layout_lines = f.readlines()

        if expression_set_type == _ExpressionSetTypeIdentifier.sensor:
            if len(layout_lines) == 0:
                # No sensor layout specified for either sensor type
                return_dict[_Keys.meg_layout] = None
                return_dict[_Keys.eeg_layout] = None
            else:
                meg_layout_parts = layout_lines[0].strip().split("\t")
                eeg_layout_parts = layout_lines[1].strip().split("\t")
                meg_identifier = meg_layout_parts[0]
                meg_layout_name = meg_layout_parts[1]
                eeg_identifier = eeg_layout_parts[0]
                eeg_layout_name = eeg_layout_parts[1]
                # The layout should be a pair of lines
                #   MEG:\t<meg layout name>
                #   EEG:\t<eeg layout name>
                # Validate that the first part of each line is as expected
                assert meg_identifier == "MEG:"
                assert eeg_identifier == "EEG:"
                # The <_ layout name> may be "None" if the layout is not specified, this is valid
                # Otherwise it should be a valid layout name
                if meg_layout_name == "None":
                    return_dict[_Keys.meg_layout] = None
                else:
                    return_dict[_Keys.meg_layout] = MEGLayout(meg_layout_name)
                if eeg_layout_name == "None":
                    return_dict[_Keys.eeg_layout] = None
                else:
                    return_dict[_Keys.eeg_layout] = EEGLayout(eeg_layout_name)

        return_dict[_Keys.channels] = dict()
        return_dict[_Keys.data] = dict()
        for block_name in blocks:
            with TextIOWrapper(zf.open(f"/{block_name}/channels.txt"), encoding="utf-8") as f:
                return_dict[_Keys.channels][block_name] = [
                    c.strip()
                    for c in f.readlines()
                ]
            with zf.open(f"/{block_name}/coo-coords.bytes") as f:
                coords: NDArray = frombuffer(f.read(), dtype=int64).reshape((3, -1))
            with zf.open(f"/{block_name}/coo-data.bytes") as f:
                data: NDArray = frombuffer(f.read(), dtype=float64)
            with TextIOWrapper(zf.open(f"/{block_name}/coo-shape.txt"), encoding="utf-8") as f:
                shape: tuple[int, ...] = tuple(int(s.strip()) for s in f.readlines())
            sparse_data = COO(coords=coords, data=data, shape=shape, prune=True, fill_value=0.0)
            # In case there was only 1 transform, and we have a 2-d data matrix
            if len(shape) == 2:
                sparse_data = expand_dims(sparse_data)
            assert shape == (
                len(return_dict[_Keys.channels][block_name]),
                len(return_dict[_Keys.latencies]),
                len(return_dict[_Keys.transforms]),
            )
            return_dict[_Keys.data][block_name] = sparse_data
    return return_dict
