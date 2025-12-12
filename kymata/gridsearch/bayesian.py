from logging import getLogger
from pathlib import Path
from typing import Optional

import os
import numpy as np
import math
from numpy.typing import NDArray, ArrayLike
from scipy import stats
from matplotlib import pyplot as plt

from kymata.entities.transform import Transform
from kymata.math.combinatorics import generate_derangement
from kymata.io.layouts import SensorLayout
from kymata.math.vector import normalize, get_stds
from kymata.entities.expression import ExpressionSet, SensorExpressionSet, HexelExpressionSet
from kymata.math.probability import LOGP_BASE, p_to_logp
from kymata.plot.gridsearch import plot_top_five_channels_of_gridsearch

_logger = getLogger(__name__)


def do_gridsearch(
        emeg_values: NDArray,  # chan x time
        transform_data: dict,
        channel_names: list,
        channel_space: str,
        start_latency: float,   # ms
        emeg_t_start: float,    # ms
        stimulus_shift_correction: float,  # seconds/second
        stimulus_delivery_latency: float,  # seconds
        plot_location: Path,
        emeg_sample_rate: float = 1000,  # Hertz
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
        transform_data (dict): The function against which the EMEG data will be correlated. It should
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
    num_functions = len(transform_data)

    '''might need to do drift correction here'''

    assumed_std_noise_of_observations = 0.5

    prior_hypothesis = np.ones(num_functions)/num_functions

    emeg_end = 800 - (-200)
    latency_step = 5
    num_latencies = len(list(range(0, emeg_end, latency_step)))

    posterior_emeg = np.zeros((n_channels, num_latencies, num_functions))

    function_mat = np.vstack(list(transform_data.values()))
    function_names = list(transform_data.keys())

    '''to discuss with kaibo: are different latencies independent between each other'''

    audio_start_correction = int(0.026 * 1000)
    '''emeg values (400300 version) drift correction (to be squeezed or stretched), try stretching/squeezing here'''

    '''drift correction'''
    original_samples = function_mat
    stretch_factor = 1 + 0.0005404
    # stretch_factor = 1 - 0.0005404
    new_length = int(function_mat.shape[1] * stretch_factor)
    original_time = np.arange(0, function_mat.shape[1] * 0.001, 0.001)
    new_time = np.linspace(original_time[0], original_time[-1], new_length)
    stretched_samples = np.zeros((num_functions, new_length))

    for i in range(num_functions):
        stretched_samples[i] = np.interp(new_time, original_time, original_samples[i])

    # things to do:
    # put drift correction into matrix format
    # try both stretch and squeeze

    '''all function posterior'''
    for latency in range(0, emeg_end, latency_step):
        '''emeg_values start from emeg_t_start (-200)'''
        print('latency: ', latency)
        emeg_values_cut = emeg_values[:, :,
                          audio_start_correction+latency: audio_start_correction+stretched_samples.shape[1]+latency]
        for channel in range(n_channels):
            single_channel = emeg_values_cut[channel][0]
            '''put my plausibility code here'''
            diff_mat = stretched_samples - single_channel
            evidence = np.log(prior_hypothesis) + \
                       -np.sum((diff_mat / (math.sqrt(2) * assumed_std_noise_of_observations)) ** 2, axis=1)  # log posterior
            single_channel_neg = - single_channel
            diff_mat_neg = stretched_samples - single_channel_neg
            evidence_neg = np.log(prior_hypothesis) + \
                       -np.sum((diff_mat_neg / (math.sqrt(2) * assumed_std_noise_of_observations)) ** 2, axis=1)
            evidence_final = np.ones(num_functions)/num_functions
            for ct in range(len(evidence_final)):  # over transforms
                if evidence[ct] >= evidence_neg[ct]:
                    evidence_final[ct] = evidence[ct]
                else:
                    evidence_final[ct] = evidence_neg[ct]
            posterior_emeg[channel, latency//latency_step] = evidence_final / (-np.sum(evidence_final))  # avoid dividing by a minus number
            # diff_mat = function_mat_x_paired - single_channel
            # evidence = np.log(prior_hypothesis_0_paired) + \
            #            -np.sum((diff_mat / (math.sqrt(2) * assumed_std_noise_of_observations)) ** 2, axis=1)
            # single_channel_neg = - single_channel
            # diff_mat_neg = function_mat_x_paired - single_channel_neg
            # evidence_neg = np.log(prior_hypothesis_0_paired) + \
            #            -np.sum((diff_mat_neg / (math.sqrt(2) * assumed_std_noise_of_observations)) ** 2, axis=1)
            # if evidence[0] >= evidence_neg[0]:
            #     evidence_final = evidence
            # else:
            #     evidence_final = evidence_neg
            # posterior_emeg[channel, latency//latency_step] = evidence_final / (-np.sum(evidence_final))

    np.save(plot_location / "posterior.npy", posterior_emeg)

    # '''load posterior'''
    # posterior_emeg = np.load('figures/posterior_emeg_all_posneg_250516.npy')

    # '''random function'''
    # mean = np.mean(stretched_samples[0])
    # std = np.mean(stretched_samples[0]**2) - mean**2
    # random_func = np.random.normal(mean, std, new_length)
    #
    # '''single function posterior'''
    # function_mat_x_paired = np.zeros((2, new_length))
    # function_mat_x_paired[0], function_mat_x_paired[1] = stretched_samples[1], random_func
    #
    # posterior_emeg = np.zeros((n_channels, num_latencies, 2))
    #
    # prior_hypothesis_0_paired = np.ones(2) / 2
    #
    # for latency in range(0, emeg_end, latency_step):
    #     '''emeg_values start from emeg_t_start (-200)'''
    #     print('latency: ', latency)
    #     emeg_values_cut = emeg_values[:, :,
    #                       audio_start_correction+latency: audio_start_correction+stretched_samples.shape[1]+latency]
    #     for channel in range(n_channels):
    #         single_channel = emeg_values_cut[channel][0]
    #         '''put my plausibility code here'''
    #         diff_mat = function_mat_x_paired - single_channel
    #         evidence = np.log(prior_hypothesis_0_paired) + \
    #                    -np.sum((diff_mat / (math.sqrt(2) * assumed_std_noise_of_observations)) ** 2, axis=1)
    #         single_channel_neg = - single_channel
    #         diff_mat_neg = function_mat_x_paired - single_channel_neg
    #         evidence_neg = np.log(prior_hypothesis_0_paired) + \
    #                    -np.sum((diff_mat_neg / (math.sqrt(2) * assumed_std_noise_of_observations)) ** 2, axis=1)
    #         if evidence[0] >= evidence_neg[0]:
    #             evidence_final = evidence
    #         else:
    #             evidence_final = evidence_neg
    #         posterior_emeg[channel, latency//latency_step] = evidence_final / (-np.sum(evidence_final))

    x_axis = np.arange(-200, 800, 5)

    # '''target vs random function comparison'''
    # evidence_of_function_across_latencies = posterior_emeg[:, :, 0]
    # random_function_across_latencies = posterior_emeg[:, :, 1]
    # for chann in range(n_channels):
    #     evidence_of_function_across_latencies_for_channel = evidence_of_function_across_latencies[chann, :]
    #     random_function_across_latencies_for_channel = random_function_across_latencies[chann, :]
    #     plt.plot(x_axis, evidence_of_function_across_latencies_for_channel, color='black', linestyle='-', linewidth=2)
    #     # plt.plot(x_axis, random_function_across_latencies_for_channel, color='grey', linestyle='-', linewidth=2)
    #
    #     # '''indicate the maximum plausibility with latency via red dot/line (see 25/01/09 screenshot)'''
    #     # max_index = np.argmax(evidence_of_function_across_latencies_for_channel)
    #     # max_x = x_axis[max_index]
    #     # max_y = evidence_of_function_across_latencies_for_channel[max_index]
    #     # min_index = np.argmin(evidence_of_function_across_latencies_for_channel)
    #     # min_x = x_axis[min_index]
    #     # min_y = evidence_of_function_across_latencies_for_channel[min_index]
    #     # plt.plot(max_x, max_y, 'ro')
    #     # # ymin, ymax = plt.ylim()
    #     # # plt.vlines(max_x, ymin, max_y, colors='r', linestyles='--')
    #     # plt.ylim(min_y, max_y)
    #     # plt.yticks(np.arange(min_y, max_y, (max_y-min_y)/10))
    #
    # plt.xlim(-200, 801)
    # plt.xticks(np.arange(-200, 801, 100))
    # plt.xlabel('latency')
    # plt.ylabel('evidence')
    # plt.title(f'plausibility across latencies for channel: {str(chann)}')
    # save_dir = './figures/stretched_func1_(vs_rand)_all_participants'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # plt.savefig(os.path.join(save_dir, str(chann) + '.png'))
    # plt.close()

    '''ensemble: each function with all channels'''
    for func in range(num_functions):
        evidence_of_function_across_latencies = posterior_emeg[:, :, func]
        for chann in range(n_channels):
            if chann == 25:
                continue
            evidence_of_function_across_latencies_for_channel = evidence_of_function_across_latencies[chann, :]
            plt.plot(x_axis, evidence_of_function_across_latencies_for_channel, color='grey', linestyle='-', linewidth=2)
            '''indicate the maximum plausibility with latency via red dot/line (see 25/01/09 screenshot)'''
            max_index = np.argmax(evidence_of_function_across_latencies_for_channel)
            max_x = x_axis[max_index]
            max_y = evidence_of_function_across_latencies_for_channel[max_index]
            plt.plot(max_x, max_y, 'ro')
            # ymin, ymax = plt.ylim()
            # plt.vlines(max_x, ymin, max_y, colors='r', linestyles='--')
        pass
        plt.xlim(-200, 801)
        plt.xticks(np.arange(-200, 801, 100))
        plt.xlabel('latency')
        plt.ylabel('evidence')

        plt.title(f'plausibility across latencies for all channel: {function_names[func]}')
        save_dir = plot_location
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, function_names[func] + '.png'))
        plt.close()

    # '''ensemble: each channel with all functions'''
    # for chann in range(n_channels):
    #     evidence_of_function_across_latencies = posterior_emeg[chann, :, :]
    #     for func in range(num_functions):
    #         evidence_of_function_across_latencies_for_channel = evidence_of_function_across_latencies[:, func]
    #         plt.plot(x_axis, evidence_of_function_across_latencies_for_channel, color='grey', linestyle='-', linewidth=2)
    #         # indicate the maximum plausibility with latency via red dot/line (see 25/01/09 screenshot)
    #         # max_index = np.argmax(evidence_of_function_across_latencies_for_channel)
    #         # max_x = x_axis[max_index]
    #         # max_y = evidence_of_function_across_latencies_for_channel[max_index]
    #         # plt.plot(max_x, max_y, 'ro')
    #         # ymin, ymax = plt.ylim()
    #         # plt.vlines(max_x, ymin, max_y, colors='r', linestyles='--')
    #     pass
    #     plt.xlim(-200, 801)
    #     plt.xticks(np.arange(-200, 801, 100))
    #     plt.xlabel('latency')
    #     plt.ylabel('evidence')
    #
    #     plt.title(f'plausibility across latencies for all function: {chann}')
    #     save_dir = './figures/each channel with all functions'
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     plt.savefig(os.path.join(save_dir, str(chann) + '.png'))
    #     plt.close()



    pass

    # int(start_latency - emeg_t_start
    #     + round(i * emeg_sample_rate * seconds_per_split * (1 + stimulus_shift_correction))  # splits, stretched by the shift correction
    #     + round(stimulus_delivery_latency * emeg_sample_rate)  # correct for stimulus delivery latency delay
    #     )


    # derive pvalues
    # evidence = '???'
    #
    # latencies_ms = np.linspace(start_latency, start_latency + (seconds_per_split * 1000), n_func_samples_per_split + 1)[:-1]
    #
    # es = '???'
    #
    # return es
