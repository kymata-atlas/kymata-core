from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
import math
from numpy.typing import NDArray, ArrayLike
from scipy import stats

from kymata.entities.functions import Function
from kymata.math.combinatorics import generate_derangement
from kymata.math.vector import normalize, get_stds
from kymata.entities.expression import ExpressionSet, SensorExpressionSet, HexelExpressionSet
from kymata.math.p_values import log_base, p_to_logp
from kymata.plot.plot import plot_top_five_channels_of_gridsearch

_logger = getLogger(__name__)


def do_gridsearch(
        emeg_values: NDArray,  # chan x time
        function_data: dict,
        channel_names: list,
        channel_space: str,
        start_latency: float,   # ms
        emeg_t_start: float,    # ms
        stimulus_shift_correction: float,  # seconds/second
        stimulus_delivery_latency: float,  # seconds
        plot_location: Optional[Path] = None,
        emeg_sample_rate: int = 1000,  # Hertz
        n_derangements: int = 1,
        seconds_per_split: float = 1,
        n_splits: int = 400,
        n_reps: int = 1,
        plot_top_five_channels: bool = False,
        overwrite: bool = True,
) -> ExpressionSet:
    """
    Perform a grid search over all hexels for all latencies using EMEG data and a given function.

    This function processes EMEG data to compute the correlation between sensor or source signals
    and a specified function across multiple latencies. The results include statistical significance
    testing and optional plotting.

    Args:
        emeg_values (NDArray): A 2D array of EMEG values with shape (n_channels, time).
        function (Function): The function against which the EMEG data will be correlated. It should
            have a `values` attribute representing the function's values and a `sample_rate`
            attribute indicating its sample rate.
        channel_names (list): List of channel names corresponding to the EMEG data. For 'sensor' space,
            it is a flat list of sensor names. For 'source' space, it is a list containing two lists:
            left hemisphere and right hemisphere hexel names.
        channel_space (str): The type of channel space used, either 'sensor' or 'source'.
        start_latency (float): The starting latency for the grid search in milliseconds.
        emeg_t_start (float): The starting time of the EMEG data in milliseconds.
        stimulus_shift_correction (float): Correction factor for stimulus shift in seconds per second.
        stimulus_delivery_latency (float): Correction offset for stimulus delivery in seconds.
        plot_location (Optional[Path], optional): Path to save the plot of the top five channels of the
            grid search. If None, plotting is skipped. Default is None.
        emeg_sample_rate (int, optional): The sample rate of the EMEG data in Hertz. Default is 1000 Hz.
        n_derangements (int, optional): Number of derangements (random permutations) used to create the
            null distribution. Default is 1.
        seconds_per_split (float, optional): Duration of each split in seconds. Default is 0.5 seconds.
        n_splits (int, optional): Number of splits used for analysis. Default is 800.
        n_reps (int, optional): Number of repetitions for each split. Default is 1.
        plot_top_five_channels (bool, optional): Plots the p-values and correlation values of the top
            five channels in the gridsearch. Default is False.
        overwrite (bool, optional): Whether to overwrite existing plot files. Default is True.

    Returns:
        ExpressionSet: An ExpressionSet object (either SensorExpressionSet or HexelExpressionSet)
        containing the log p-values for each channel/hexel and latency.

    Notes:
        - The function down-samples the EMEG data to match the function's sample rate.
        - The EMEG data is reshaped into segments of the specified duration (`seconds_per_split`).
        - Cross-correlations between the EMEG data and the function are computed using FFT.
        - Statistical significance is assessed using a vectorized Welch's t-test.
        - If specified, the results are plotted and saved to the given location.
    """

    # Set random seed to keep derangement orderings
    # deterministic between runs
    np.random.seed(17)

    '''set function priors, 1/number_of_functions'''

    channel_space = channel_space.lower()
    if channel_space not in {"sensor", "source"}:
        raise NotImplementedError(channel_space)

    n_channels = emeg_values.shape[0]
    num_functions = len(function_data)

    '''currently we do this simple cut, later we consider audio latency and drift correction'''
    emeg_values = emeg_values[:, :, :400000]

    assumed_std_noise_of_observations = 0.5

    prior_hypothesis = np.ones(num_functions)/num_functions

    posterior_emeg = np.zeros((n_channels, num_functions))


    for channel in range(n_channels):
        single_channel = emeg_values[channel][0]
        '''put my plausibility code here'''
        function_mat = np.vstack(list(function_data.values()))
        diff_mat = function_mat - single_channel
        evidence = np.log(prior_hypothesis) + \
                   -np.sum((diff_mat / (math.sqrt(2) * assumed_std_noise_of_observations)) ** 2, axis=1)
        posterior_emeg[channel] = evidence / np.sum(evidence)

    pass

    # int(start_latency - emeg_t_start
    #     + round(i * emeg_sample_rate * seconds_per_split * (1 + stimulus_shift_correction))  # splits, stretched by the shift correction
    #     + round(stimulus_delivery_latency * emeg_sample_rate)  # correct for stimulus delivery latency delay
    #     )


    # derive pvalues
    evidence = '???'

    latencies_ms = np.linspace(start_latency, start_latency + (seconds_per_split * 1000), n_func_samples_per_split + 1)[:-1]

    # if plot_top_five_channels:
    #     plot_top_five_channels_of_gridsearch(
    #         corrs=corrs,
    #         auto_corrs=auto_corrs,
    #         function=function,
    #         n_reps=n_reps,
    #         n_splits=n_splits,
    #         n_samples_per_split=n_samples_per_split,
    #         latencies=latencies_ms,
    #         save_to=plot_location,
    #         log_pvalues=log_pvalues,
    #         overwrite=overwrite,
    #     )
    #
    # if channel_space == "sensor":
    #     es = SensorExpressionSet(
    #         functions=function.name,
    #         latencies=latencies_ms / 1000,  # seconds
    #         sensors=channel_names,
    #         data=log_pvalues,
    #     )
    # elif channel_space == "source":
    #     es = HexelExpressionSet(
    #         functions=function.name,
    #         latencies=latencies_ms / 1000,  # seconds
    #         hexels_lh=channel_names[0],
    #         hexels_rh=channel_names[1],
    #         # Unstack the data
    #         data_lh=log_pvalues[:len(channel_names[0]), :],
    #         data_rh=log_pvalues[len(channel_names[0]):, :],
    #     )
    # else:
    #     raise NotImplementedError(channel_space)

    es = '???'
    return es
