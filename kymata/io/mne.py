from pathlib import Path

from mne import read_evokeds
from numpy import max
from numpy.typing import NDArray

from kymata.io.file import path_type


def get_emeg_data(emeg_path: path_type) -> tuple[NDArray, list[str]]:
    """Gets EMEG data from MNE format."""
    emeg_path = Path(emeg_path)

    evoked = read_evokeds(emeg_path, verbose=False)  # should be len 1 list
    emeg = evoked[0].get_data()  # numpy array shape (sensor_num, N) = (370, 403_001)

    # Normalise
    emeg /= max(emeg, axis=1, keepdims=True)
    return emeg, evoked[0].ch_names
