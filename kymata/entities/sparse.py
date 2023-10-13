from __future__ import annotations

from numpy import ndarray, nanmin, greater
import sparse
from xarray import Dataset


def expand_dims(x: sparse.COO, axis=-1) -> sparse.COO:
    """
    Expand the dims of a sparse.COO matrix by inserting a size-1 dim at the specified axis.
    axis=-1 means to add the dim at the end.
    Raises a ValueError if x isn't a 2-dimensional matrix
    """
    shape = list(x.shape)
    if len(shape) > 3:
        raise ValueError("Too many dimensions")
    if len(shape) == 3:
        if shape[2] != 1:
            raise ValueError("Too many dimensions")
        # Already expanded
        return x
    if axis == -1:
        shape.append(1)
    else:
        shape[axis:axis] = [1]
    x = x.reshape(tuple(shape))
    return x


def minimise_pmatrix(pmatrix: ndarray) -> sparse.COO:
    """
    Converts a hexel-x-latency data matrix containing p-values into a sparse matrix
    only storing the minimum (over latencies) p-value for each hexel.
    """
    hexel_min = nanmin(pmatrix, axis=1)
    hexel_min = expand_dims(hexel_min, axis=2)
    # Minimum p-val over all latencies for this hexel
    pmatrix[greater(pmatrix, hexel_min)] = 1.0
    return sparse.COO(pmatrix, fill_value=1.0)


def densify_dataset(x: Dataset):
    """
    Converts data in an xarray.Dataset wrapping a sparse.COO matrix to a dense numpy array.
    Operates in-place.
    """
    for var in x.data_vars.keys():
        x[var].data = x[var].data.todense()
