from enum import StrEnum

from packaging import version
from io import TextIOWrapper
from pathlib import Path
from typing import Any
from warnings import warn
from zipfile import ZipFile, ZIP_LZMA

from numpy import ndarray, frombuffer
from sparse import COO

from kymata.entities.datatypes import HexelDType, LatencyDType, FunctionNameDType, SensorDType
from kymata.entities.expression import ExpressionSet, LAYER_LEFT, LAYER_RIGHT, LAYER_SCALP, HexelExpressionSet, \
    SensorExpressionSet, p_to_logp
from kymata.entities.sparse_data import expand_dims
from kymata.io.file import path_type, file_type, open_or_use
from kymata.io.nkg_legacy import _load_data_0_1, _load_data_0_2


class _Keys(StrEnum):
    channels  = "channels"
    latencies = "latencies"
    functions = "functions"
    layers    = "layers"
    data      = "data"

    expressionset_type = "expressionset-type"


class _ExpressionSetTypeIdentifier(StrEnum):
    hexel  = "hexel"
    sensor = "sensor"

    @classmethod
    def from_expression_set(cls, expression_set: ExpressionSet):
        if isinstance(expression_set, HexelExpressionSet):
            return cls.hexel
        elif isinstance(expression_set, SensorExpressionSet):
            return cls.sensor
        else:
            raise NotImplementedError()


# Format versioning of the saved file. Used to (eventually) ensure old saved files can always be loaded.
#
# - Increment MINOR versions when creating a breaking change, but only when we commit to indefinitely supporting
#   the new version.
# - Increment MAJOR version when creating a breaking change which makes the format fundamentally incompatible
#   with previous versions.
#
# All distinct versions should be documented.
#
# This value should be saved into file called /_metadata/format-version.txt within an archive.
CURRENT_VERSION = "0.3"


def file_version(from_path_or_file: path_type | file_type) -> version.Version:
    with open_or_use(from_path_or_file, mode="rb") as archive, ZipFile(archive, "r") as zf:
        with TextIOWrapper(zf.open("_metadata/format-version.txt"), encoding="utf-8") as f:
            return version.parse(str(f.read()).strip())


# TODO: Could maybe improve this using generics when dropping support for Python version <3.12
def load_expression_set(from_path_or_file: path_type | file_type) -> ExpressionSet:
    _v, data_dict = _load_data(from_path_or_file)

    type_identifier = data_dict[_Keys.expressionset_type]

    if type_identifier == _ExpressionSetTypeIdentifier.hexel:
        return HexelExpressionSet(
            functions=data_dict[_Keys.functions],
            hexels=[HexelDType(c) for c in data_dict[_Keys.channels]],
            latencies=data_dict[_Keys.latencies],
            data_lh=[data_dict[_Keys.data][LAYER_LEFT][:, :, i]
                     for i in range(len(data_dict[_Keys.functions]))],
            data_rh=[data_dict[_Keys.data][LAYER_RIGHT][:, :, i]
                     for i in range(len(data_dict[_Keys.functions]))],
        )
    elif type_identifier == _ExpressionSetTypeIdentifier.sensor:
        return SensorExpressionSet(
            functions=data_dict[_Keys.functions],
            sensors=[SensorDType(c) for c in data_dict[_Keys.channels]],
            latencies=data_dict[_Keys.latencies],
            data=[data_dict[_Keys.data][LAYER_SCALP][:, :, i]
                  for i in range(len(data_dict[_Keys.functions]))],
        )


def save_expression_set(expression_set: ExpressionSet,
                        to_path_or_file: path_type | file_type,
                        compression=ZIP_LZMA,
                        overwrite: bool = False):
    """
    Save the ExpressionSet to a specified path or already open file.

    If an open file is supplied, it should be opened in "wb" mode.

    overwrite flag is ignored if open file is supplied.
    """

    warn("Experimental function. "
         "The on-disk data format for ExpressionSet is not yet fixed. "
         "Files saved using .save should not (yet) be treated as stable or future-proof.")

    if isinstance(to_path_or_file, str):
        to_path_or_file = Path(to_path_or_file)
    if isinstance(to_path_or_file, Path) and to_path_or_file.exists() and not overwrite:
        raise FileExistsError(to_path_or_file)

    with open_or_use(to_path_or_file, mode="wb") as f, ZipFile(f, "w", compression=compression) as zf:
        zf.writestr("_metadata/format-version.txt", CURRENT_VERSION)
        zf.writestr("_metadata/expression-set-type.txt", _ExpressionSetTypeIdentifier.from_expression_set(expression_set))
        zf.writestr("/channels.txt",  "\n".join(str(x) for x in expression_set._channels))
        zf.writestr("/latencies.txt", "\n".join(str(x) for x in expression_set.latencies))
        zf.writestr("/functions.txt", "\n".join(str(x) for x in expression_set.functions))
        zf.writestr("/layers.txt",    "\n".join(str(x) for x in expression_set._layers))

        for layer in expression_set._layers:
            zf.writestr(f"/{layer}/coo-coords.bytes", expression_set._data[layer].data.coords.tobytes(order="C"))
            zf.writestr(f"/{layer}/coo-data.bytes", expression_set._data[layer].data.data.tobytes(order="C"))
            # The shape can be inferred, but we save it as an extra validation
            zf.writestr(f"/{layer}/coo-shape.txt", "\n".join(str(x) for x in expression_set._data[layer].data.shape))


def _load_data(from_path_or_file: path_type | file_type) -> tuple[version.Version, dict[str, Any]]:
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
        dict_0_1 = _load_data_0_1(from_path_or_file)

        # v0.1 data was stored as p values
        data = dict()
        for layer, sparse_data_dict in dict_0_1["data"].items():
            sparse_data_dict["data"] = p_to_logp(sparse_data_dict["data"])

            sparse_data = COO(coords=sparse_data_dict["coords"],
                              data=sparse_data_dict["data"],
                              shape=sparse_data_dict["shape"],
                              prune=True, fill_value=0.0)
            assert sparse_data.shape == (
                len(dict_0_1["hexels"]),
                len(dict_0_1["latencies"]),
                len(dict_0_1["functions"]),
            )
            # In case there was only 1 function and we have a 2-d data matrix
            if len(sparse_data.shape) == 2:
                sparse_data = expand_dims(sparse_data)
            data[layer] = sparse_data

        return_dict = {
            _Keys.channels:  dict_0_1["hexels"],
            _Keys.functions: dict_0_1["functions"],
            _Keys.latencies: dict_0_1["latencies"],
            _Keys.data:      data,
            # Keys not present in v0.1
            _Keys.expressionset_type: _ExpressionSetTypeIdentifier.hexel,
        }
    elif v <= version.parse("0.2"):
        dict_0_2 = _load_data_0_2(from_path_or_file)

        # v0.2 data was stored as p-values
        data = dict()
        for layer, sparse_data_dict in dict_0_2["data"].items():
            sparse_data_dict["data"] = p_to_logp(sparse_data_dict["data"])

            sparse_data = COO(coords=sparse_data_dict["coords"],
                              data=sparse_data_dict["data"],
                              shape=sparse_data_dict["shape"],
                              prune=True, fill_value=0.0)
            assert sparse_data.shape == (
                len(dict_0_2["channels"]),
                len(dict_0_2["latencies"]),
                len(dict_0_2["functions"]),
            )
            # In case there was only 1 function and we have a 2-d data matrix
            if len(sparse_data.shape) == 2:
                sparse_data = expand_dims(sparse_data)
            data[layer] = sparse_data
        return_dict = {
            _Keys.expressionset_type: dict_0_2["expressionset-type"],
            _Keys.channels:           dict_0_2["channels"],
            _Keys.functions:          dict_0_2["functions"],
            _Keys.latencies:          dict_0_2["latencies"],
            _Keys.data:               data,
        }
    else:
        return_dict = _load_data_current(from_path_or_file)

    return v, return_dict


# noinspection DuplicatedCode
def _load_data_current(from_path_or_file: path_type | file_type) -> dict[str, Any]:
    """
    Load data from current version
    """
    return_dict = dict()
    with open_or_use(from_path_or_file, mode="rb") as archive, ZipFile(archive, "r") as zf:

        with TextIOWrapper(zf.open("_metadata/expression-set-type.txt"), encoding="utf-8") as f:
            return_dict[_Keys.expressionset_type] = str(f.read()).strip()
        with TextIOWrapper(zf.open("/layers.txt"), encoding="utf-8") as f:
            layers = [str(l.strip()) for l in f.readlines()]
        with TextIOWrapper(zf.open("/channels.txt"), encoding="utf-8") as f:
            return_dict[_Keys.channels] = [c.strip() for c in f.readlines()]
        with TextIOWrapper(zf.open("/latencies.txt"), encoding="utf-8") as f:
            return_dict[_Keys.latencies] = [LatencyDType(lat.strip()) for lat in f.readlines()]
        with TextIOWrapper(zf.open("/functions.txt"), encoding="utf-8") as f:
            return_dict[_Keys.functions] = [FunctionNameDType(fun.strip()) for fun in f.readlines()]
        return_dict[_Keys.data] = dict()
        for layer in layers:
            with zf.open(f"/{layer}/coo-coords.bytes") as f:
                coords: ndarray = frombuffer(f.read(), dtype=int).reshape((3, -1))
            with zf.open(f"/{layer}/coo-data.bytes") as f:
                data: ndarray = frombuffer(f.read(), dtype=float)
            with TextIOWrapper(zf.open(f"/{layer}/coo-shape.txt"), encoding="utf-8") as f:
                shape: tuple[int, ...] = tuple(int(s.strip()) for s in f.readlines())
            sparse_data = COO(coords=coords, data=data, shape=shape, prune=True, fill_value=1.0)
            # In case there was only 1 function and we have a 2-d data matrix
            if len(shape) == 2:
                sparse_data = expand_dims(sparse_data)
            assert shape == (len(return_dict[_Keys.channels]),
                             len(return_dict[_Keys.latencies]),
                             len(return_dict[_Keys.functions]))
            return_dict[_Keys.data][layer] = sparse_data
    return return_dict
