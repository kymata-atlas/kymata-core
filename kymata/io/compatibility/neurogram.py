from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kymata.entities.contours import ContourSet
from kymata.io.file import PathType


def load_neurogram_npz(
        location: PathType,
        transform_name: str,
        bruce_neurons: tuple = (0, 10),
) -> ContourSet:
    trans_dict = np.load(location)
    trans = trans_dict[transform_name]
    trans = np.mean(trans[bruce_neurons[0] : bruce_neurons[1]], axis=0)
    return ContourSet(
        contour_dict={transform_name: trans},
        sample_rate_hz=1000,  # This is assumed by these legacy filetypes
    )

def load_neurogram_mat(
        location: PathType,
        transform_name: str,
        bruce_neurons: tuple = (0, 10),
) -> ContourSet:
    """Load neurogram transforms from a (legacy) .mat file."""
    location = Path(location)

    mat = loadmat(location)["neurogramResults"]
    trans = np.array(mat[transform_name][0][0])
    # trans = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)
    tt = np.array(mat["t_" + transform_name[-2:]][0][0])
    n_chans = trans.shape[0]
    new_mr_arr = np.zeros((n_chans, 400_000))
    new_ft_arr = np.zeros((n_chans, 400_000))
    if transform_name == "neurogram_mr":
        for i in range(n_chans):
            new_mr_arr[i] = np.interp(
                np.linspace(0, 400, 400_000 + 1)[:-1], tt[0], trans[i]
            )
    elif transform_name == "neurogram_ft":
        trans = np.cumsum(trans * (tt[0, 1] - tt[0, 0]), axis=-1)
        for i in range(n_chans):
            trans_interp = np.interp(
                np.linspace(0, 400, 400_000 + 1), tt[0], trans[i]
            )
            new_ft_arr[i] = np.diff(trans_interp, axis=-1)

    trans_dict = {"neurogram_mr": new_mr_arr, "neurogram_ft": new_ft_arr}

    trans = trans_dict[transform_name]
    trans = np.mean(trans[bruce_neurons[0]: bruce_neurons[1]], axis=0)

    return ContourSet(
        contour_dict={transform_name: trans},
        sample_rate_hz=1000,  # This is assumed by these legacy filetypes
    )
