from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import sys
sys.path.append('/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox')

from kymata.gridsearch.gridsearch import do_gridsearch
from kymata.io.functions import load_function
from kymata.io.mne import get_emeg_data
from kymata.plot.plotting import expression_plot


def main():
    downsample_rate = 5
    function_name = 'd_IL2'

    base_dir = Path("/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/")

    func = load_function(Path(base_dir,
                              'predicted_function_contours',
                              'GMSloudness',
                              'stimulisig'),
                         func_name=function_name)
    func = func.downsampled(downsample_rate)

    emeg_dir = Path(base_dir,
                    "intrim_preprocessing_files",
                    "3_trialwise_sensorspace",
                    "evoked_data")
    emeg_filename = 'participant_01-ave'

    # Load data
    emeg_path_npy = Path(emeg_dir, f"{emeg_filename}.npy")
    emeg_path_fif = Path(emeg_dir, f"{emeg_filename}.fif")
    try:
        ch_names: list[str] = []  # TODO: we'll need these
        emeg: NDArray = np.load(emeg_path_npy)
    except FileNotFoundError:
        emeg, ch_names = get_emeg_data(emeg_path_fif)
        np.save(emeg_path_npy, np.array(emeg, dtype=np.float16))

    es = do_gridsearch(
        emeg_values=emeg,
        sensor_names=ch_names,
        function=func,
        seconds_per_split=0.5,
        n_derangements=1,
        n_splits=800,
        start_latency=-100,
        emeg_t_start=-200,
    )

    expression_plot(es)


if __name__ == '__main__':
    main()
