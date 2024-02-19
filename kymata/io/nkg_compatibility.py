"""
# Version 0.4: allowed mismatched hexel ids in the left and right hemispheres
#
# Version 0.3: moved to log p-values.
#
# Version 0.2: supports sensor data.
#
# Version 0.1: original layered sparse data format.
"""

from io import TextIOWrapper
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from numpy import frombuffer
from numpy.typing import NDArray

from kymata.entities.datatypes import LatencyDType, FunctionNameDType, HexelDType
from kymata.io.file import path_type, file_type, open_or_use


# noinspection DuplicatedCode
def _load_data_0_3(from_path_or_file: path_type | file_type) -> dict[str, Any]:
    """
    This is a function which loads data format 0.3.

    The idea is that *this function never changes*, and handling code instead can change to
    transform the returned dictionary to whatever format is required for the current
    iteration of the appropriate classes.

    In particular, that means it uses inline string keys rather than keys stored in the Keys class.
    """
    return_dict = dict()
    with open_or_use(from_path_or_file, mode="rb") as archive, ZipFile(archive, "r") as zf:

        with TextIOWrapper(zf.open("_metadata/expression-set-type.txt"), encoding="utf-8") as f:
            return_dict["expressionset-type"] = str(f.read()).strip()
        with TextIOWrapper(zf.open("/layers.txt"), encoding="utf-8") as f:
            layers = [str(line.strip()) for line in f.readlines()]
        with TextIOWrapper(zf.open("/channels.txt"), encoding="utf-8") as f:
            return_dict["channels"] = [c.strip() for c in f.readlines()]
        with TextIOWrapper(zf.open("/latencies.txt"), encoding="utf-8") as f:
            return_dict["latencies"] = [LatencyDType(lat.strip()) for lat in f.readlines()]
        with TextIOWrapper(zf.open("/functions.txt"), encoding="utf-8") as f:
            return_dict["functions"] = [FunctionNameDType(fun.strip()) for fun in f.readlines()]
        return_dict["data"] = dict()
        for layer in layers:
            with zf.open(f"/{layer}/coo-coords.bytes") as f:
                coords: NDArray = frombuffer(f.read(), dtype=int).reshape((3, -1))
            with zf.open(f"/{layer}/coo-data.bytes") as f:
                data: NDArray = frombuffer(f.read(), dtype=float)
            with TextIOWrapper(zf.open(f"/{layer}/coo-shape.txt"), encoding="utf-8") as f:
                shape: tuple[int, ...] = tuple(int(s.strip()) for s in f.readlines())
            return_dict["data"][layer] = dict()
            return_dict["data"][layer]["coords"] = coords
            return_dict["data"][layer]["shape"] = shape
            return_dict["data"][layer]["data"] = data
    return return_dict


# noinspection DuplicatedCode
def _load_data_0_2(from_path_or_file: path_type | file_type) -> dict[str, Any]:
    """
    This is a function which loads data format 0.2.

    The idea is that *this function never changes*, and handling code instead can change to
    transform the returned dictionary to whatever format is required for the current
    iteration of the appropriate classes.

    In particular, that means it uses inline string keys rather than keys stored in the Keys class.
    """
    return_dict = dict()
    with open_or_use(from_path_or_file, mode="rb") as archive, ZipFile(archive, "r") as zf:

        with TextIOWrapper(zf.open("_metadata/expression-set-type.txt"), encoding="utf-8") as f:
            return_dict["expressionset-type"] = str(f.read()).strip()
        with TextIOWrapper(zf.open("/layers.txt"), encoding="utf-8") as f:
            layers = [str(line.strip()) for line in f.readlines()]
        with TextIOWrapper(zf.open("/channels.txt"), encoding="utf-8") as f:
            return_dict["channels"] = [c.strip() for c in f.readlines()]
        with TextIOWrapper(zf.open("/latencies.txt"), encoding="utf-8") as f:
            return_dict["latencies"] = [LatencyDType(lat.strip()) for lat in f.readlines()]
        with TextIOWrapper(zf.open("/functions.txt"), encoding="utf-8") as f:
            return_dict["functions"] = [FunctionNameDType(fun.strip()) for fun in f.readlines()]
        return_dict["data"] = dict()
        for layer in layers:
            with zf.open(f"/{layer}/coo-coords.bytes") as f:
                coords: NDArray = frombuffer(f.read(), dtype=int).reshape((3, -1))
            with zf.open(f"/{layer}/coo-data.bytes") as f:
                data: NDArray = frombuffer(f.read(), dtype=float)
            with TextIOWrapper(zf.open(f"/{layer}/coo-shape.txt"), encoding="utf-8") as f:
                shape: tuple[int, ...] = tuple(int(s.strip()) for s in f.readlines())
            return_dict["data"][layer] = dict()
            return_dict["data"][layer]["coords"] = coords
            return_dict["data"][layer]["shape"] = shape
            return_dict["data"][layer]["data"] = data
    return return_dict


# noinspection DuplicatedCode
def _load_data_0_1(from_path_or_file: path_type | file_type) -> dict[str, Any]:
    """
    This is a function which loads data format 0.1.

    The idea is that *this function never changes*, and handling code instead can change to
    transform the returned dictionary to whatever format is required for the current
    iteration of the appropriate classes.

    In particular, that means it uses inline string keys rather than keys stored in the Keys class.
    """

    if isinstance(from_path_or_file, str):
        from_path_or_file = Path(from_path_or_file)

    return_dict = dict()
    with open_or_use(from_path_or_file, mode="rb") as archive, ZipFile(archive, "r") as zf:
        with TextIOWrapper(zf.open("/hexels.txt"), encoding="utf-8") as f:
            return_dict["hexels"]: list[HexelDType] = [HexelDType(h.strip()) for h in f.readlines()]
        with TextIOWrapper(zf.open("/latencies.txt"), encoding="utf-8") as f:
            return_dict["latencies"]: list[LatencyDType] = [LatencyDType(lat.strip()) for lat in f.readlines()]
        with TextIOWrapper(zf.open("/functions.txt"), encoding="utf-8") as f:
            return_dict["functions"]: list[FunctionNameDType] = [FunctionNameDType(fun.strip()) for fun in f.readlines()]
        return_dict["data"] = dict()
        for layer in ["left", "right"]:
            with zf.open(f"/{layer}/coo-coords.bytes") as f:
                coords: NDArray = frombuffer(f.read(), dtype=int).reshape((3, -1))
            with zf.open(f"/{layer}/coo-data.bytes") as f:
                data: NDArray = frombuffer(f.read(), dtype=float)
            with TextIOWrapper(zf.open(f"/{layer}/coo-shape.txt"), encoding="utf-8") as f:
                shape: tuple[int, ...] = tuple(int(s.strip()) for s in f.readlines())
            return_dict["data"][layer] = dict()
            return_dict["data"][layer]["coords"] = coords
            return_dict["data"][layer]["shape"] = shape
            return_dict["data"][layer]["data"] = data

    return return_dict
