from os import path

from numpy import array, float16
from numpy.typing import NDArray
from h5py import File
from scipy.io import loadmat

from kymata.entities.functions import Function


def load_function(function_path: str, func_name: str) -> Function:
    if not path.isfile(function_path + '.h5'):
        mat = loadmat(function_path + '.mat')['stimulisig']
        with File(function_path + '.h5', 'w') as f:
            for key in mat.dtype.names:
                if key != 'name':
                    f.create_dataset(key, data=array(mat[key][0, 0], dtype=float16))
        func: NDArray = array(mat[func_name][0][0])
    else:
        with File(function_path + '.h5', 'r') as f:
            func: NDArray = f[func_name]

    return Function(
        name=func_name,
        values=func.flatten().squeeze(),
        tstep=0.001,
    )
