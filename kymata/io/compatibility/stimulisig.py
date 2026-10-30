from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kymata.entities.contours import ContourSet
from kymata.io.file import PathType


def load_stimulisig_mat(location: PathType) -> ContourSet:
    """
    Loads a (legacy) <stimulisig>.mat file as a ContourSet.
    """
    location = Path(location)

    mat = loadmat(str(location))["stimulisig"]
    trans_dict = {}
    for key in mat.dtype.names:
        if key == "name":
            continue
        trans_dict[key] = np.array(mat[key][0, 0], dtype=np.float16)
        trans_dict[key].reshape((1, -1))  # Unwrap if it's a split matlab stimulisig

    return ContourSet(contour_dict=trans_dict)
