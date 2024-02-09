from __future__ import annotations

from numpy import nanmin, greater
from numpy.typing import NDArray
import sparse
from xarray import DataArray


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


def sparsify_log_pmatrix(log_pmatrix: NDArray) -> sparse.COO:
    """
    Converts a channel-x-latency data matrix containing log p-values into a sparse matrix
    only storing the minimum (over latencies) log p-value for each channel.
    """
    fill_value = 0.0
    channel_min = nanmin(log_pmatrix, axis=1)
    channel_min = expand_dims(channel_min, axis=2)
    # Minimum value over all latencies for this channel
    log_pmatrix[greater(log_pmatrix, channel_min)] = fill_value
    return sparse.COO(log_pmatrix, fill_value=fill_value)


def densify_data_block(x: DataArray) -> None:
    """
    Converts data in an xarray.DataArray wrapping a sparse.COO matrix to a dense numpy array.
    Operates in-place.
    """
    x.data = x.data.todense()
