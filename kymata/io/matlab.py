from scipy.io import loadmat as loadmat_pre_73
from mat73 import loadmat as loadmat_post_73


def load_mat(path):
    """Loads all variables in a matlab file, regardless of version."""
    try:
        with path.open("rb") as f:
            mat = loadmat_pre_73(f, appendmat=False)
    except NotImplementedError:
        mat = loadmat_post_73(path)
    return mat


def load_mat_variable(path, variable_name: str):
    return load_mat(path)[variable_name]
