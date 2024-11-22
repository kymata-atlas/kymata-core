from abc import ABC
from copy import deepcopy, copy
from random import shuffle
from statistics import NormalDist
from typing import Optional

from .cluster import (
    MaxPoolClusterer, AdaptiveMaxPoolClusterer, GMMClusterer, DBSCANClusterer, MeanShiftClusterer, CustomClusterer,
    ANOMALOUS_CLUSTER_TAG)
from .build import SpikeDict
from ..entities.expression import ExpressionPoint
from ..math.p_values import logp_to_p, p_to_logp


class DenoisingStrategy(ABC):
    """
    Superclass for unsupervised clustering algorithms.
    Strategies should conform to this interface.
    """

    def __init__(
        self,
        should_normalise: bool = False,
        should_cluster_only_latency: bool = False,
        should_max_pool: bool = False,
        normal_dist_threshold: float = 5,
        should_exclude_insignificant: bool = True,
        should_shuffle: bool = True,
        **kwargs,
    ):
        """
        :param should_normalise:
                Indicates whether the preprocessing step of scaling the data to be unit length should be done.
                Unnormalised data places greater precedence on the latency dimension.
        :param cluster_only_latency:
                Indicates whether we should discard the magnitude dimension from the clustering algorithm.
                We can term this "density-based clustering" as it focuses on the density of points in time only.
        :param should_max_pool:
                Indicates whether we want to enforce the constraint that each transform should have 1 spike.
                Equivalently, we take the maximum over all clusters and discard all other points. Setting this to
                True optimises performance in the context of IPPMs.
        :param should_exclude_insignificant:
                Indicates whether we want to exclude insignificant spikes before clustering.
        """

        self._clusterer: CustomClusterer = None
        self._should_normalise = should_normalise
        self._should_cluster_only_latency = should_cluster_only_latency
        self._should_max_pool = should_max_pool
        self._should_exclude_insignificant = should_exclude_insignificant
        self._should_shuffle = should_shuffle
        self._normal_dist_threshold = normal_dist_threshold

    @staticmethod
    def _estimate_threshold_for_significance(x: float, n_timepoints: int, n_hexels: int) -> float:
        """
        Bonferroni corrected = probability / number_of_trials.

        :param x: indicates the probability of P(X < x) for NormalDist(0, 1)
        :returns: a float indicating that threshold for significance.
        """

        def __calculate_bonferroni_corrected_alpha(one_minus_probability):
            return 1 - pow(
                (1 - one_minus_probability), (1 / (2 * n_timepoints * n_hexels))
            )

        probability_X_gt_x = 1 - NormalDist(mu=0, sigma=1).cdf(x)
        threshold_for_significance = __calculate_bonferroni_corrected_alpha(
            probability_X_gt_x
        )
        return threshold_for_significance

    def denoise(self, spikes: SpikeDict) -> SpikeDict:
        """
        For a set of transforms, cluster their IPPMSpike and retain the most significant spikes per cluster.

        Algorithm
        ---------
        For each transform,
            1. We create a list of ExpressionPairings containing only significant spikes.
                1.1) [Optional] Perform any preprocessing.
                    - should_normalise: Scale each dimension to have a total length of 1.
                    - cluster_only_latency: Remove magnitude dimension.
            2. Cluster using the clustering method of the child class.
            3. For each cluster, we take the most significant point and discard the rest.
                3.1) [Optional] Perform any postprocessing.
                    - should_max_pool: Take the most significant point over all the clusters. I.e., only 1
                                             spike per transform.

        :param spikes:
            The key is the transform name and the IPPMSpike contains information about the transform. Specifically,
            it contains a list of (Latency (ms), Magnitude (^10-x)) for each hemisphere. We cluster over one
            hemisphere.
        :return: A dictionary of transforms as keys and a IPPMSpike containing the clustered time-series.
        """
        for func, points in spikes.items():

            points = copy(points)
            # Apply filters
            if self._should_exclude_insignificant:
                points = self._filter_out_insignificant_pairings(points)
            if self._should_shuffle:
                # shuffle to remove correlations between rows
                shuffle(points)


            if len(points) == 0:
                spikes[func] = points
                continue

            preprocessed_points = self._preprocess(points)

            self._clusterer: CustomClusterer = self._clusterer.fit(preprocessed_points)
            # It is important you don't use the preprocessed_df to get the denoised because
            # we do not want to plot the preprocessed latencies and magnitudes.
            spikes[func] = self._get_denoised_time_series(points)

            if self._should_max_pool:
                spikes[func] = self._perform_max_pooling(spikes[func])

            self._clusterer.reset()

        return spikes

    def _filter_out_insignificant_pairings(self, points: list[ExpressionPoint]) -> list[ExpressionPoint]:
        """
        For a list of ExpressionPoints, remove all that are not statistically significant and return a list of those
        that remain.

        :param points: pairings we want to filter based on their statistical significance.
        :returns: a new list containing pairings with significant p-values.
        """
        return [
            p for p in points
            if logp_to_p(p.logp_value) <= self._estimate_threshold_for_significance(
                self._normal_dist_threshold,
                n_timepoints=n_timepoints,
                n_hexels=n_hexels
            )
        ]

    def _preprocess(self, points: list[ExpressionPoint]) -> list[ExpressionPoint]:
        """
        Currently, this can normalise or remove the magnitude dimension.

        :param points: points we want to preprocess.
        :return: if we cluster_only_latency, we return a numpy array. Else, a list of ExpressionPoint.
        """

        points = deepcopy(points)
        if self._should_normalise:
            """
            Short text on normalisation for future reference.
            https://www.kaggle.com/code/residentmario/l1-norms-versus-l2-norms
            """
            points = self._normalize(points, self._should_cluster_only_latency)
        return points
    
    @staticmethod
    def _normalize(points: list[ExpressionPoint], latency_only: bool) -> list[ExpressionPoint]:
        # Normalise latency
        latency_sum = sum(p.latency for p in points)
        if latency_only:
            pval_sum = 1
        else:
            pval_sum = sum(logp_to_p(p.logp_value) for p in points)

        return [
            ExpressionPoint(
                channel=p.channel,
                latency=p.latency / latency_sum,
                transform=p.transform,
                logp_value=p_to_logp(logp_to_p(p.logp_value) / pval_sum))
            for p in points
        ]

    def _get_denoised_time_series(self, points: list[ExpressionPoint]) -> list[ExpressionPoint]:
        """
        For a given list of ExpressionPoint, extract out the most significant spike for each cluster along with its
        associated latency and ignore anomalies.

        Assumes self._clusterer.fit() has been called already.

        :param points: list of points for the time series.
        :returns: list of pairings (denoised).
        """

        def __keep_most_significant_per_label(points_: list[ExpressionPoint],
                                              labels: list[int],
                                              ) -> tuple[list[ExpressionPoint],
                                                         list[int]]:
            from pandas import DataFrame

            assert len(points_) == len(labels)

            df = DataFrame({
                "logp":  [p.logp_value for p in points_],
                "label":   labels,
            })
            idxs = df.groupby("label")["logp"].idxmin()

            return (
                [points_[i] for i in idxs],  # Filtered points
                [labels[i] for i in idxs],   # Filtered labels
            )

        points, labels = __keep_most_significant_per_label(points, self._clusterer.labels_)
        # Filter out anomalies
        points = [point
                  for point, label in zip(points, labels)
                  if label != ANOMALOUS_CLUSTER_TAG]
        return points

    def _perform_max_pooling(self, spike: list[ExpressionPoint]) -> list[ExpressionPoint]:
        """
        Enforce the constraint that there is only 1 spike for a specific transform. It is basically max(all_spikes).

        :param spike: Contains the time-series we want to take the maximum over.
        :returns: same spike but with only one spike.
        """
        # we take minimum because smaller is more significant.
        return [
            min(spike, key=lambda x: x.logp_value)
        ]


class MaxPoolingStrategy(DenoisingStrategy):
    def __init__(self,
                 should_normalise: bool = True,
                 should_cluster_only_latency: bool = False,
                 should_max_pool: bool = False,
                 normal_dist_threshold: float = 5,
                 should_exclude_insignificant: bool = True,
                 should_shuffle: bool = True,
                 bin_significance_threshold: int = 1,
                 bin_size: int = 1,
    ):
        super().__init__(
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
            should_exclude_insignificant,
            should_shuffle
        )
        self._clusterer = MaxPoolClusterer(bin_significance_threshold, bin_size)


class AdaptiveMaxPoolingStrategy(DenoisingStrategy):
    def __init__(self,
                 should_normalise: bool = True,
                 should_cluster_only_latency: bool = False,
                 should_max_pool: bool = False,
                 normal_dist_threshold: float = 5,
                 should_exclude_insignificant: bool = True,
                 bin_significance_threshold: int = 1,
                 base_bin_size: int = 1,
    ):
        super().__init__(
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
            should_exclude_insignificant,
            should_shuffle=False,  # AMP assumes a sorted time series, so avoid shuffling.
        )                          # Mainly because AMP uses a sliding window algorithm to merge significant clusters.
        self._clusterer = AdaptiveMaxPoolClusterer(bin_significance_threshold, base_bin_size)


class GMMStrategy(DenoisingStrategy):
    def __init__(self,
                 should_normalise: bool = True,
                 should_cluster_only_latency: bool = True,
                 should_max_pool: bool = False,
                 normal_dist_threshold: float = 5,
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
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
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
    def __init__(self,
                 should_normalise: bool = False,
                 should_cluster_only_latency: bool = False,
                 should_max_pool: bool = False,
                 normal_dist_threshold: float = 5,
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
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
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
    def __init__(self,
                 should_normalise: bool = True,
                 should_cluster_only_latency: bool = True,
                 should_max_pool: bool = False,
                 normal_dist_threshold: float = 5,
                 should_exclude_insignificant: bool = True,
                 should_shuffle: bool = True,
                 cluster_all: bool = False,
                 bandwidth: float = 0.5,
                 seeds: Optional[int] = None,
                 min_bin_freq: int = 1,
                 n_jobs: int = -1,
    ):
        super().__init__(
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            normal_dist_threshold,
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
