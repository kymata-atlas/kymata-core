from contextlib import contextmanager
from pathlib import Path
from typing import TextIO, BinaryIO, Union

file_type = Union[TextIO, BinaryIO]
path_type = Union[str, Path]


@contextmanager
def open_or_use(path_or_file: path_type | file_type, mode: str = "r") -> file_type:
    """
    If passed a path, will open it and return the file handle, and close when done.
    if passed a file handle, will keep it open, and return it when done.

    Use like:

        with open_or_use(path) as f:
            ...

    or like:

        with open(path) as f:
            with open_or_use(f) as ff:
                ...

    (Thanks to https://stackoverflow.com/a/6783680/2883198)
    """
    if isinstance(path_or_file, str):
        path_or_file = Path(path_or_file)
    if isinstance(path_or_file, Path):
        f = file_to_close = path_or_file.open(mode)
    else:
        f = path_or_file
        file_to_close = None

    try:
        yield f
    finally:
        if file_to_close:
            file_to_close.close()
