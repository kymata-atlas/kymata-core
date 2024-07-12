from copy import deepcopy
from statistics import NormalDist
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.preprocessing import normalize

from .constants import TIMEPOINTS, NUMBER_OF_HEXELS
from .custom_clusterers import MaxPooler, AdaptiveMaxPooler, CustomGMM
from .data_tools import IPPMHexel


class DenoisingStrategy(object):
    """Superclass for unsupervised clustering algorithms. Strategies should conform to this interface."""

    def __init__(
            self,
            hemi: str,
            should_normalise: bool = False,
            should_cluster_only_latency: bool = False,
            should_max_pool: bool = False,
            normal_dist_threshold: float = 5,
            **kwargs
    ):
        """
        :param hemi: Either "rightHemisphere" or "leftHemisphere". Indicates which hemisphere to cluster.
        :param should_normalise:
                Indicates whether the preprocessing step of scaling the data to be unit length should be done.
                Unnormalised data places greater precedence on the latency dimension.
        :param cluster_only_latency:
                Indicates whether we should discard the magnitude dimension from the clustering algorithm.
                We can term this "density-based clustering" as it focuses on the density of points in time only.
        :param should_max_pool:
                Indicates whether we want to enforce the constraint that each function should have 1 spike.
                Equivalently, we take the maximum over all clusters and discard all other points. Setting this to
                True optimises performance in the context of IPPMs.
        """

        self._clusterer = None
        self._hemi = hemi
        self._should_normalise = should_normalise
        self._should_cluster_only_latency = should_cluster_only_latency
        self._should_max_pool = should_max_pool
        self._threshold_for_significance = DenoisingStrategy._estimate_threshold_for_significance(normal_dist_threshold)

    @staticmethod
    def _estimate_threshold_for_significance(x: float) -> float:
        """
        bonferroni corrected = probability / number_of_trials.

        :param x: indicates the probability of P(X < x) for NormalDist(0, 1)
        :returns: a float indicating that threshold for significance.
        """

        def __calculate_bonferroni_corrected_alpha(one_minus_probability):
            return 1 - pow((1 - one_minus_probability), (1 / (2 * TIMEPOINTS * NUMBER_OF_HEXELS)))

        probability_X_gt_x = 1 - NormalDist(mu=0, sigma=1).cdf(x)
        threshold_for_significance = __calculate_bonferroni_corrected_alpha(probability_X_gt_x)
        return threshold_for_significance

    def denoise(self, hexels: Dict[str, IPPMHexel]) -> Dict[str, IPPMHexel]:
        """
            For a set of functions, cluster their IPPMHexel and retain the most significant spikes per cluster.

            TODO: Isn't max pooling after clustering equivalent to max pooling before? Only difference is that
                  anomalies are excluded. In that case, we could just MAX(all_spikes) to get optimal strategy.

            TODO: Figure out how to manage side-effects

            Algorithm
            ---------
            For each function in hemi,
                1. We create a dataframe containing only significant spikes.
                    1.1) [Optional] Perform any preprocessing.
                        - should_normalise: Scale each dimension to have a total length of 1.
                        - cluster_only_latency: Remove magnitude dimension.
                2. Cluster using the clustering method of the child class.
                3. For each cluster, we take the most significant point and discard the rest.
                    3.1) [Optional] Perform any postprocessing.
                        - should_max_pool: Take the most significant point over all of the clusters. I.e., only 1
                                                 spike per function.

            :param hexels:
                The key is the function name and the IPPMHexel contains information about the function. Specifically,
                it contains a list of (Latency (ms), Magnitude (^10-x)) for each hemisphere. We cluster over one
                hemisphere.
            :return: A dictionary of functions as keys and a IPPMHexel containing the clustered time-series.
        """
        hexels = deepcopy(
            hexels)  # When we copy the input, it is because we don't want to modify the original structure.
        # These are called side effects, and they can introduce obscure bugs.

        for func, df in self._map_hexels_to_df(hexels):
            if len(df) == 0:
                hexels[func] = self._update_pairings(hexels[func], [])
                continue

            preprocessed_df = self._preprocess(df)

            self._clusterer = self._clusterer.fit(preprocessed_df)
            # It is important you don't use the preprocessed_df to get the denoised because
            # we do not want to plot the preprocessed latencies and magnitudes.
            denoised_time_series = self._get_denoised_time_series(df)

            hexels[func] = self._postprocess(hexels[func], denoised_time_series)

        return hexels

    def _map_hexels_to_df(self, hexels: Dict[str, IPPMHexel]) -> pd.DataFrame:
        """
        A generator used to transform each pair of key, IPPMHexel to a DataFrame containing significance spikes only.

        :param hexels: The dictionary we want to iterate over and transform into DataFrames.
        :returns: a clean DataFrame with columns ['Latency', 'Mag'] for each function key.
        """
        for func, hexel in hexels.items():
            significant_spikes = self._filter_out_insignificant_spikes(
                hexel.right_best_pairings if self._hemi == 'rightHemisphere' else hexel.left_best_pairings,
            )
            df = pd.DataFrame(significant_spikes, columns=['Latency', 'Mag'])
            yield func, df

    def _filter_out_insignificant_spikes(self, spikes: List[Tuple[float, float]]) -> List[List[float]]:
        """
        For a list of spikes, remove all that are not statistically significant and save them to a DataFrame.

        :param spikes: spikes we want to filter based on their statistical significance.
        :param df: dataframe we will save the spikes to. Assumption: first column is Latency, second is Mag.
        :returns: DataFrame that is a subset of spikes with only significant spikes.
        """
        significant_datapoints = []
        for latency, spike in spikes:
            if spike <= self._threshold_for_significance:
                significant_datapoints.append([latency, spike])
        return significant_datapoints

    def _update_pairings(self, hexel: IPPMHexel, denoised: List[Tuple[float, float]]) -> IPPMHexel:
        """
        :param hexel: We want to update this hexel to store the denoised spikes, with max pooling if desired.
        :param denoised: We want to save these spikes into hexel, overwriting the previous ones.
        :returns: IPPM hexel with its state updated.
        """
        if self._hemi == 'rightHemisphere':
            hexel.right_best_pairings = denoised
        else:
            hexel.left_best_pairings = denoised
        return hexel

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Currently, this can normalise or remove the magnitude dimension.

        :param df: Dataframe we want to preprocess.
        :return: if we cluster_only_latency, we return a numpy array. Else, dataframe.
        """

        def __extract_and_wrap_latency_dim(df_with_latency_col):
            return np.reshape(df_with_latency_col['Latency'], (-1, 1))

        mags = list(df["Mag"])
        if self._should_cluster_only_latency:
            df = __extract_and_wrap_latency_dim(df)
        if self._should_normalise:
            # Assumption: we only have latency, mag columns.
            """
                Short text on normalisation for future reference.
                https://www.kaggle.com/code/residentmario/l1-norms-versus-l2-norms
            """
            normed_latency_and_mag = np.c_[normalize(df, axis=0), mags]
            df = pd.DataFrame(normed_latency_and_mag, columns=["Latency", "Mag"])
        return df

    def _get_denoised_time_series(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        """
        For a given dataframe, extract out the most significant spike for each cluster along with its associated latency
        and ignore anomalies (label == -1.)

        Assumes self._clusterer.fit has been called already. Also, assumes anomalies are tagged as "-1".

        :param df: Contains cols ["Latency", "Mag"] and IPPMHexel time-series.
        :returns: Each tuple contains ("Latency", "Mag") of the most significant points (excl. anomalies).
        """

        def __keep_most_significant_per_label(labelled_df):
            return labelled_df.loc[labelled_df.groupby('Label')['Mag'].idxmin()]

        def __filter_out_anomalies(labelled_df):
            return labelled_df[labelled_df['Label'] != -1]

        def __convert_df_to_list(most_significant_points_df):
            return list(zip(most_significant_points_df['Latency'], most_significant_points_df['Mag']))

        df = deepcopy(df)
        df["Label"] = self._clusterer.labels_
        most_significant_points = __keep_most_significant_per_label(df)
        most_significant_points = __filter_out_anomalies(most_significant_points)
        return __convert_df_to_list(most_significant_points)

    def _postprocess(self, hexel: IPPMHexel, denoised_time_series: List[Tuple[float, float]]) -> IPPMHexel:
        """
        To postprocess, overwrite the hexel data with most significant points and perform any postprocessing steps,
        such as max pooling.

        :param hexel: We will overwrite the data in this hexel with denoised data.
        :param denoised_time_series: most significant points per cluster excluding anomalies.
        :return: hexel with denoised time-series saved and, optionally, max pooled.
        """
        hexel = self._update_pairings(hexel, denoised_time_series)
        if self._should_max_pool:
            hexel = self._perform_max_pooling(hexel)
        return hexel

    def _perform_max_pooling(self, hexel: IPPMHexel) -> IPPMHexel:
        """
        Enforce the constraint that there is only 1 spike for a specific function. It is basically max(all_spikes).

        :param hexel: Contains the time-series we want to take the maximum over.
        :returns: same hexel but with only one spike.
        """
        # we take minimum because smaller is more significant.
        if hexel.left_best_pairings and self._hemi == 'leftHemisphere':
            hexel.left_best_pairings = [min(hexel.left_best_pairings, key=lambda x: x[1])]
        elif hexel.right_best_pairings and self._hemi == 'rightHemisphere':
            hexel.right_best_pairings = [min(hexel.right_best_pairings, key=lambda x: x[1])]
        return hexel


class MaxPoolingStrategy(DenoisingStrategy):
    def __init__(
            self,
            hemi: str,
            should_normalise: bool = False,
            should_cluster_only_latency: bool = False,
            should_max_pool: bool = False,
            normal_dist_threshold: float = 5,
            bin_significance_threshold: int = 15,
            bin_size: int = 25
    ):
        super().__init__(hemi, should_normalise, should_cluster_only_latency, should_max_pool, normal_dist_threshold)
        self._clusterer = MaxPooler(bin_significance_threshold, bin_size)


class AdaptiveMaxPoolingStrategy(DenoisingStrategy):
    def __init__(
            self,
            hemi: str,
            should_normalise: bool = False,
            should_cluster_only_latency: bool = False,
            should_max_pool: bool = False,
            normal_dist_threshold: float = 5,
            bin_significance_threshold: int = 5,
            base_bin_size: int = 10
    ):
        super().__init__(hemi, should_normalise, should_cluster_only_latency, should_max_pool, normal_dist_threshold)
        self._clusterer = AdaptiveMaxPooler(bin_significance_threshold, base_bin_size)


class GMMStrategy(DenoisingStrategy):
    def __init__(
            self,
            hemi: str,
            should_normalise: bool = False,
            should_cluster_only_latency: bool = False,
            should_max_pool: bool = False,
            normal_dist_threshold: float = 5,
            number_of_clusters_upper_bound: int = 5,
            covariance_type: str = 'full',
            max_iter: int = 1000,
            n_init: int = 8,
            init_params: str = 'kmeans',
            random_state: Optional[int] = None,
            should_evaluate_using_AIC: bool = False
    ):
        super().__init__(hemi, should_normalise, should_cluster_only_latency, should_max_pool, normal_dist_threshold)
        self._clusterer = CustomGMM(number_of_clusters_upper_bound, covariance_type, max_iter, n_init, init_params,
                                    random_state, should_evaluate_using_AIC)


class DBSCANStrategy(DenoisingStrategy):
    def __init__(
            self,
            hemi: str,
            should_normalise: bool = False,
            should_cluster_only_latency: bool = False,
            should_max_pool: bool = False,
            normal_dist_threshold: float = 5,
            eps: int = 10,
            min_samples: int = 2,
            metric: str = 'euclidean',
            algorithm: str = 'auto',
            leaf_size: int = 30,
            n_jobs: int = -1,
            metric_params: Optional[dict] = None
    ):
        super().__init__(hemi, should_normalise, should_cluster_only_latency, should_max_pool, normal_dist_threshold)
        self._clusterer = DBSCAN(eps=eps,
                                 min_samples=min_samples,
                                 metric=metric,
                                 metric_params=metric_params,
                                 algorithm=algorithm,
                                 leaf_size=leaf_size,
                                 n_jobs=n_jobs)


class MeanShiftStrategy(DenoisingStrategy):
    def __init__(
            self,
            hemi: str,
            should_normalise: bool = False,
            should_cluster_only_latency: bool = False,
            should_max_pool: bool = False,
            normal_dist_threshold: float = 5,
            cluster_all: bool = False,
            bandwidth: float = 30,
            seeds: Optional[int] = None,
            min_bin_freq: int = 2,
            n_jobs: int = -1
    ):
        super().__init__(hemi, should_normalise, should_cluster_only_latency, should_max_pool, normal_dist_threshold)
        self._clusterer = MeanShift(bandwidth=bandwidth,
                                    seeds=seeds,
                                    min_bin_freq=min_bin_freq,
                                    cluster_all=cluster_all,
                                    n_jobs=n_jobs)
