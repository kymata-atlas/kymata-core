from abc import ABC
from copy import deepcopy
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

from .constants import TIMEPOINTS, NUMBER_OF_HEXELS
from .cluster import (
    MaxPoolClusterer, AdaptiveMaxPoolClusterer, GMMClusterer, DBSCANClusterer, MeanShiftClusterer, CustomClusterer)
from .data_tools import IPPMSpike, SpikeDict, ExpressionPairing
from ..entities.constants import HEMI_RIGHT, HEMI_LEFT


# Column names
LATENCY = "Latency"
MAGNITUDE = "Magnitude"


class DenoisingStrategy(ABC):
    """
    Superclass for unsupervised clustering algorithms.
    Strategies should conform to this interface.
    """

    def __init__(
        self,
        hemi: str,
        should_normalise: bool = False,
        should_cluster_only_latency: bool = False,
        should_max_pool: bool = False,
        normal_dist_threshold: float = 5,
        should_merge_hemis: bool = False,
        should_exclude_insignificant: bool = True,
        should_shuffle: bool = True,
        **kwargs,
    ):
        """
        :param hemi: Either "right" or "left". Indicates which hemisphere to cluster.
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
        :param should_merge_hemis:
                Indicates whether we want to combine the two hemispheres into one, then cluster. If this is True, then
                the final expression plot is plot overwrites the "hemi" expression plot. E.g., if hemi == HEMI_RIGHT, then overwrite it
        :param should_exclude_insignificant:
                Indicates whether we want to exclude insignificant spikes before clustering.
        """

        self._clusterer: CustomClusterer = None
        self._hemi = hemi
        self._should_normalise = should_normalise
        self._should_cluster_only_latency = should_cluster_only_latency
        self._should_max_pool = should_max_pool
        self._should_merge_hemis = should_merge_hemis
        self._should_exclude_insignificant = should_exclude_insignificant
        self._should_shuffle = should_shuffle
        self._threshold_for_significance = (
            DenoisingStrategy._estimate_threshold_for_significance(
                normal_dist_threshold
            )
        )

    @staticmethod
    def _estimate_threshold_for_significance(x: float) -> float:
        """
        Bonferroni corrected = probability / number_of_trials.

        :param x: indicates the probability of P(X < x) for NormalDist(0, 1)
        :returns: a float indicating that threshold for significance.
        """

        def __calculate_bonferroni_corrected_alpha(one_minus_probability):
            return 1 - pow(
                (1 - one_minus_probability), (1 / (2 * TIMEPOINTS * NUMBER_OF_HEXELS))
            )

        probability_X_gt_x = 1 - NormalDist(mu=0, sigma=1).cdf(x)
        threshold_for_significance = __calculate_bonferroni_corrected_alpha(
            probability_X_gt_x
        )
        return threshold_for_significance

    def denoise(self, spikes: SpikeDict) -> SpikeDict:
        """
        For a set of functions, cluster their IPPMSpike and retain the most significant spikes per cluster.

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

        :param spikes:
            The key is the function name and the IPPMSpike contains information about the function. Specifically,
            it contains a list of (Latency (ms), Magnitude (^10-x)) for each hemisphere. We cluster over one
            hemisphere.
        :return: A dictionary of functions as keys and a IPPMSpike containing the clustered time-series.
        """
        spikes = deepcopy(
            spikes
        )  # When we copy the input, it is because we don't want to modify the original structure.
        # These are called side effects, and they can introduce obscure bugs.

        for func, df in self._map_spikes_to_df(spikes):
            if len(df) == 0:
                spikes[func] = self._update_pairings(spikes[func], [])
                continue

            preprocessed_df = self._preprocess(df)

            self._clusterer = self._clusterer.fit(preprocessed_df)
            # It is important you don't use the preprocessed_df to get the denoised because
            # we do not want to plot the preprocessed latencies and magnitudes.
            denoised_time_series = self._get_denoised_time_series(df)

            spikes[func] = self._postprocess(spikes[func], denoised_time_series)

            self._clusterer.labels_ = []  # reset clusterer labels between runs

        return spikes

    def _map_spikes_to_df(self, spikes: SpikeDict) -> pd.DataFrame:
        """
        A generator used to transform each pair of key, IPPMSpike to a DataFrame containing significance spikes only.

        :param spikes: The dictionary we want to iterate over and transform into DataFrames.
        :returns: a clean DataFrame with columns ['Latency', 'Mag'] for each function key.
        """
        for func, spike in spikes.items():
            extracted_spikes = spike.right_best_pairings if self._hemi == HEMI_RIGHT else spike.left_best_pairings
            if self._should_merge_hemis:
                other_hemi_spikes = spike.left_best_pairings if self._hemi == HEMI_LEFT else spike.right_best_pairings
                extracted_spikes.extend(other_hemi_spikes)
            if self._should_exclude_insignificant:
                extracted_spikes = self._filter_out_insignificant_spikes(
                    extracted_spikes
                    if self._hemi == HEMI_RIGHT
                    else spike.left_best_pairings,
                )
            df = pd.DataFrame(extracted_spikes, columns=[LATENCY, MAGNITUDE])
            if self._should_shuffle:
                df = df.sample(frac=1).reset_index(drop=True)  # shuffle to remove correlations between rows.
            yield func, df

    def _filter_out_insignificant_spikes(
        self, spikes: list[ExpressionPairing]
    ) -> list[list[float]]:
        """
        For a list of spikes, remove all that are not statistically significant and save them to a DataFrame.

        :param spikes: spikes we want to filter based on their statistical significance.
        :returns: DataFrame that is a subset of spikes with only significant spikes.
        """
        significant_datapoints = []
        for latency, spike in spikes:
            if spike <= self._threshold_for_significance:
                significant_datapoints.append([latency, spike])
        return significant_datapoints

    def _update_pairings(
        self, spike: IPPMSpike, denoised: list[tuple[float, float]]
    ) -> IPPMSpike:
        """
        :param spike: We want to update this spike to store the denoised spikes, with max pooling if desired.
        :param denoised: We want to save these spikes into spike, overwriting the previous ones.
        :returns: IPPMSpike with its state updated.
        """
        if self._hemi == HEMI_RIGHT:
            spike.right_best_pairings = denoised
            if self._should_merge_hemis:
                spike.left_best_pairings = []
        else:
            spike.left_best_pairings = denoised
            if self._should_merge_hemis:
                spike.right_best_pairings = []
        return spike

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Currently, this can normalise or remove the magnitude dimension.

        :param df: Dataframe we want to preprocess.
        :return: if we cluster_only_latency, we return a numpy array. Else, dataframe.
        """

        def __extract_and_wrap_latency_dim(df_with_latency_col):
            np_arr = np.reshape(df_with_latency_col[LATENCY], (-1, 1))
            return pd.DataFrame(np_arr, columns=[LATENCY])

        df = deepcopy(df)
        if self._should_cluster_only_latency:
            df = __extract_and_wrap_latency_dim(df)
        if self._should_normalise:
            # Assumption: we only have latency, mag columns.
            """
                Short text on normalisation for future reference.
                https://www.kaggle.com/code/residentmario/l1-norms-versus-l2-norms
            """
            if self._should_cluster_only_latency:
                df = self._normalize(df, [LATENCY])
            else:
                df = self._normalize(df, [LATENCY, MAGNITUDE])
        return df
    
    @staticmethod
    def _normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            df[col] = df[col] / df[col].sum()
        return df

    def _get_denoised_time_series(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        """
        For a given dataframe, extract out the most significant spike for each cluster along with its associated latency
        and ignore anomalies (label == -1).

        Assumes self._clusterer.fit has been called already. Also, assumes anomalies are tagged as "-1".

        :param df: Contains cols ["Latency", "Mag"] and IPPMHexel time-series.
        :returns: Each tuple contains ("Latency", "Mag") of the most significant points (excl. anomalies).
        """

        def __keep_most_significant_per_label(labelled_df):
            return labelled_df.loc[labelled_df.groupby("Label")[MAGNITUDE].idxmin()]

        def __filter_out_anomalies(labelled_df):
            return labelled_df[labelled_df["Label"] != -1]

        def __convert_df_to_list(most_significant_points_df):
            return list(
                zip(
                    most_significant_points_df[LATENCY],
                    most_significant_points_df[MAGNITUDE],
                )
            )

        df = deepcopy(df)
        df["Label"] = self._clusterer.labels_
        most_significant_points = __keep_most_significant_per_label(df)
        most_significant_points = __filter_out_anomalies(most_significant_points)
        return __convert_df_to_list(most_significant_points)

    def _postprocess(
        self, spike: IPPMSpike, denoised_time_series: list[tuple[float, float]]
    ) -> IPPMSpike:
        """
        To postprocess, overwrite the spike data with most significant points and perform any postprocessing steps,
        such as max pooling.

        :param spike: We will overwrite the data in this spike with denoised data.
        :param denoised_time_series: most significant points per cluster excluding anomalies.
        :return: spike with denoised time-series saved and, optionally, max pooled.
        """
        spike = self._update_pairings(spike, denoised_time_series)
        if self._should_max_pool:
            spike = self._perform_max_pooling(spike)
        return spike

    def _perform_max_pooling(self, spike: IPPMSpike) -> IPPMSpike:
        """
        Enforce the constraint that there is only 1 spike for a specific function. It is basically max(all_spikes).

        :param spike: Contains the time-series we want to take the maximum over.
        :returns: same spike but with only one spike.
        """
        # we take minimum because smaller is more significant.
        if spike.left_best_pairings and self._hemi == HEMI_LEFT:
            spike.left_best_pairings = [
                min(spike.left_best_pairings, key=lambda x: x[1])
            ]
        elif spike.right_best_pairings and self._hemi == HEMI_RIGHT:
            spike.right_best_pairings = [
                min(spike.right_best_pairings, key=lambda x: x[1])
            ]
        return spike


class MaxPoolingStrategy(DenoisingStrategy):
    def __init__(
        self,
        hemi: str,
        should_normalise: bool = True,
        should_cluster_only_latency: bool = False,
        should_max_pool: bool = False,
        normal_dist_threshold: float = 5,
        should_merge_hemis: bool = True,
        should_exclude_insignificant: bool = True,
        should_shuffle: bool = True,
        bin_significance_threshold: int = 1,
        bin_size: int = 1,
    ):
        super().__init__(
            hemi,
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
            should_merge_hemis,
            should_exclude_insignificant,
            should_shuffle
        )
        self._clusterer = MaxPoolClusterer(bin_significance_threshold, bin_size)


class AdaptiveMaxPoolingStrategy(DenoisingStrategy):
    def __init__(
        self,
        hemi: str,
        should_normalise: bool = True,
        should_cluster_only_latency: bool = False,
        should_max_pool: bool = False,
        normal_dist_threshold: float = 5,
        should_merge_hemis: bool = True,
        should_exclude_insignificant: bool = True,
        bin_significance_threshold: int = 1,
        base_bin_size: int = 1,
    ):
        super().__init__(
            hemi,
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
            should_merge_hemis,
            should_exclude_insignificant,
            should_shuffle=False,  # AMP assumes a sorted time series, so avoid shuffling.
        )                          # Mainly because AMP uses a sliding window algorithm to merge significant clusters.
        self._clusterer = AdaptiveMaxPoolClusterer(bin_significance_threshold, base_bin_size)


class GMMStrategy(DenoisingStrategy):
    def __init__(
        self,
        hemi: str,
        should_normalise: bool = True,
        should_cluster_only_latency: bool = True,
        should_max_pool: bool = False,
        normal_dist_threshold: float = 5,
        should_merge_hemis: bool = True,
        should_exclude_insignificant: bool = True,
        should_shuffle: bool = True,
        number_of_clusters_upper_bound: int = 2,
        covariance_type: str = "full",
        max_iter: int = 1000,
        n_init: int = 5,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        should_evaluate_using_AIC: bool = True,
    ):
        super().__init__(
            hemi,
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
            should_merge_hemis,
            should_exclude_insignificant,
            should_shuffle
        )
        self._clusterer = GMMClusterer(
            number_of_clusters_upper_bound,
            covariance_type,
            max_iter,
            n_init,
            init_params,
            random_state,
            should_evaluate_using_AIC,
        )


class DBSCANStrategy(DenoisingStrategy):
    def __init__(
        self,
        hemi: str,
        should_normalise: bool = False,
        should_cluster_only_latency: bool = False,
        should_max_pool: bool = False,
        normal_dist_threshold: float = 5,
        should_merge_hemis: bool = False,
        should_exclude_insignificant: bool = True,
        should_shuffle: bool = True,
        eps: int = 10,
        min_samples: int = 2,
        metric: str = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 30,
        n_jobs: int = -1,
        metric_params: Optional[dict] = None,
    ):
        super().__init__(
            hemi,
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
            should_merge_hemis,
            should_exclude_insignificant,
            should_shuffle
        )
        self._clusterer = DBSCANClusterer(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        )


class MeanShiftStrategy(DenoisingStrategy):
    def __init__(
        self,
        hemi: str,
        should_normalise: bool = True,
        should_cluster_only_latency: bool = True,
        should_max_pool: bool = False,
        normal_dist_threshold: float = 5,
        should_merge_hemis: bool = True,
        should_exclude_insignificant: bool = True,
        should_shuffle: bool = True,
        cluster_all: bool = False,
        bandwidth: float = 0.5,
        seeds: Optional[int] = None,
        min_bin_freq: int = 1,
        n_jobs: int = -1,
    ):
        super().__init__(
            hemi,
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
            should_merge_hemis,
            should_exclude_insignificant,
            should_shuffle
        )
        self._clusterer = MeanShiftClusterer(
            bandwidth=bandwidth,
            seeds=seeds,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            n_jobs=n_jobs,
        )
