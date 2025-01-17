from abc import ABC
from copy import deepcopy, copy
from random import shuffle
from typing import Optional, overload, Callable

from ..math.probability import logp_to_p, p_to_logp, sidak_correct, p_threshold_for_sigmas
from .cluster import (
    MaxPoolClusterer, AdaptiveMaxPoolClusterer, GMMClusterer, DBSCANClusterer, MeanShiftClusterer, CustomClusterer,
    ANOMALOUS_CLUSTER_TAG)
from ..entities.expression import (
    ExpressionPoint, HexelExpressionSet, SensorExpressionSet, ExpressionSet, get_n_channels)
from .hierarchy import group_points_by_transform, GroupedPoints


class DenoisingStrategy(ABC):
    """
    Abstract base class for unsupervised clustering algorithms used in denoising expression sets.

    Strategies should conform to this interface.

    This class defines the common interface and functionality for denoising strategies that use clustering
    methods to remove noise from expression data, focusing on significant spikes and clustering points
    based on latency or other relevant dimensions.
    """

    def __init__(
        self,
        should_normalise: bool,
        should_cluster_only_latency: bool,
        should_max_pool: bool,
        exclude_logp_vals_above: float | None,
        exclude_points_above_n_sigma: float | None,
        should_shuffle: bool,
        **kwargs,
    ):
        """
        Initializes a DenoisingStrategy object with the specified parameters.

        Args:
            should_normalise (bool): Indicates whether the preprocessing step of scaling the data to be unit length
                should be done. Unnormalised data places greater precedence on the latency dimension.
            should_cluster_only_latency (bool): Indicates whether we should discard the magnitude dimension from the
                clustering algorithm. We can term this "density-based clustering" as it focuses on the density of points
                in time only.
            should_max_pool (bool): Indicates whether we want to enforce the constraint that each transform should have
                1 spike.
                Equivalently, we take the maximum over all clusters and discard all other points. Setting this to True
                optimises performance in the context of IPPMs.
            exclude_logp_vals_above (float | None): Supply a log p-value threshold to test for significance for this
                exclusion. Supply None to not exclude.
            exclude_points_above_n_sigma (float | None): Sigma threshold for excluding points.  Supply None to not
                exclude.
            should_shuffle (bool): Whether to shuffle the points before clustering.
        """

        if exclude_logp_vals_above is not None and exclude_points_above_n_sigma is not None:
            raise ValueError("Supply only one of `exclude_logp_vals_above` and `exclude_points_above_n_sigma`.")
        if exclude_points_above_n_sigma is not None:
            def get_threshold(es: ExpressionSet) -> float:
                """Get Šidák n-sigma threshold"""
                n_comparisons = len(es.transforms) * len(es.latencies) + get_n_channels(es)
                return p_to_logp(sidak_correct(p_threshold_for_sigmas(exclude_points_above_n_sigma),
                                               n_comparisons=n_comparisons))
        else:
            # Will return either the fixed threshold, or constant None where exclude_logp_vals_above was None
            def get_threshold(_es: ExpressionSet) -> float:
                """Get fixed threshold"""
                return exclude_logp_vals_above

        self._clusterer: CustomClusterer
        self._should_normalise = should_normalise
        self._should_cluster_only_latency = should_cluster_only_latency
        self._should_max_pool = should_max_pool
        self._logp_threshold_from_expression_set: Callable[[ExpressionSet], float | None] = get_threshold
        self._exclude_logp_vals_above = exclude_logp_vals_above
        self._should_shuffle = should_shuffle

    @overload
    def denoise(self, expression_set: HexelExpressionSet) -> HexelExpressionSet:
        """Denoise a HexelExpressionSet."""
        ...

    @overload
    def denoise(self, expression_set: SensorExpressionSet) -> SensorExpressionSet:
        """Denoise a SensorExpressionSet."""
        ...

    @overload
    def denoise(self, expression_set: ExpressionSet) -> ExpressionSet:
        """Denoise a general ExpressionSet."""
        ...

    def denoise(self, expression_set: ExpressionSet) -> ExpressionSet:
        """
        Denoises an expression set by applying the appropriate denoising method based on the type of the expression set.

        Args:
            expression_set (ExpressionSet): The expression set to denoise.

        Returns:
            ExpressionSet: The denoised expression set.
        """
        if isinstance(expression_set, HexelExpressionSet):
            return self._denoise_hexel_expression_set(expression_set)
        elif isinstance(expression_set, SensorExpressionSet):
            return self._denoise_sensor_expression_set(expression_set)
        else:
            raise NotImplementedError()

    def _denoise_sensor_expression_set(self, expression_set: SensorExpressionSet) -> SensorExpressionSet:
        expression_set = copy(expression_set)
        # Get the denoised spikes from the expression set's data
        expression_points = expression_set.best_transforms()
        original_spikes = group_points_by_transform(expression_points)
        denoised_spikes = self._denoise_spikes(original_spikes, self._logp_threshold_from_expression_set(expression_set))

        for transform in expression_set.transforms:
            denoised_points = denoised_spikes.get(transform, [])
            # For any points which didn't make it to the denoised set, set their logp values to 0
            for point in original_spikes[transform]:
                if point in denoised_points:
                    continue
                expression_set.clear_point(point.channel, point.latency)

        return expression_set

    def _denoise_hexel_expression_set(self, expression_set: HexelExpressionSet) -> HexelExpressionSet:
        expression_set = copy(expression_set)
        expression_points_left, expression_points_right = expression_set.best_transforms()
        original_spikes_left = group_points_by_transform(expression_points_left)
        original_spikes_right = group_points_by_transform(expression_points_right)
        denoised_spikes_left = self._denoise_spikes(original_spikes_left, self._logp_threshold_from_expression_set(expression_set))
        denoised_spikes_right = self._denoise_spikes(original_spikes_right, self._logp_threshold_from_expression_set(expression_set))

        for transform in expression_set.transforms:
            denoised_points_left = denoised_spikes_left.get(transform, [])
            denoised_points_right = denoised_spikes_right.get(transform, [])
            for point in original_spikes_left[transform]:
                # For any points which didn't make it to the denoised set, set their logp values to 0
                if point in denoised_points_left:
                    continue
                expression_set.clear_point_left(point.channel, point.latency)
            for point in original_spikes_right[transform]:
                if point in denoised_points_right:
                    continue
                expression_set.clear_point_right(point.channel, point.latency)

        return expression_set

    def _denoise_spikes(self, spikes: GroupedPoints, logp_threshold: float | None) -> GroupedPoints:
        """
        For a set of transforms, cluster their significant points and retain the most significant point per cluster.

        Algorithm
        ---------
        For each transform,
            1. We create a list of ExpressionPoints containing only significant points.
                1.1) [Optional] Perform any preprocessing.
                    - should_normalise: Scale each dimension to have a total length of 1.
                    - cluster_only_latency: Remove magnitude dimension.
            2. Cluster using the clustering method of the child class.
            3. For each cluster, we take the most significant point and discard the rest.
                3.1) [Optional] Perform any postprocessing.
                    - should_max_pool: Take the most significant point over all the clusters. I.e., only 1
                                             spike per transform.

        Args:
            spikes (GroupedPoints): A dictionary of ExpressionPoints grouped by transform name.
            logp_threshold (float | None): The log p-value threshold for excluding insignificant points.

        Returns:
            GroupedPoints: A dictionary of ExpressionPoints grouped by transform name, containing the clustered
                time-series.
        """

        spikes = deepcopy(spikes)

        for trans, points in spikes.items():

            points = copy(points)
            # Apply filters
            if logp_threshold is not None:
                points = self._filter_out_insignificant_points(points, logp_threshold)
            if self._should_shuffle:
                # shuffle to remove correlations between rows
                shuffle(points)

            if len(points) == 0:
                spikes[trans] = points
                continue

            # We use the preprocessed points to fit the clusterer, but it is important not to use them to actually get
            # the denoised points because the latencies and logp values have been altered
            #
            # Short text on normalisation for future reference.
            # https://www.kaggle.com/code/residentmario/l1-norms-versus-l2-norms
            self._clusterer: CustomClusterer = self._clusterer.fit(self._preprocess(points))

            spikes[trans] = self._get_denoised_time_series(points)
            if self._should_max_pool:
                spikes[trans] = self._perform_max_pooling(spikes[trans])

            self._clusterer.reset()

        return spikes

    @staticmethod
    def _filter_out_insignificant_points(points: list[ExpressionPoint], threshold_logp: float) -> list[ExpressionPoint]:
        """
        For a list of ExpressionPoints, remove all that are not statistically significant and return a list of those
        that remain.

        Args:
            points (list[ExpressionPoint]): Points we want to filter based on their statistical significance.
            threshold_logp (float): The log p-value threshold below which points are considered significant.

        Returns:
            list[ExpressionPoint]: A new list containing points with significant log p-values.
        """
        return [
            p for p in points
            # Lower is better
            if logp_to_p(p.logp_value) < threshold_logp
        ]

    def _preprocess(self, points: list[ExpressionPoint]) -> list[ExpressionPoint]:
        """
        Currently, this can normalise or remove the magnitude dimension.

        Args:
            points (list[ExpressionPoint]): Points we want to preprocess.

        Returns:
            list[ExpressionPoint]: The preprocessed ExpressionPoints.
                NOTE that these points are NOT true representations of the original data — they can be used for
                clustering only but should not be interpreted directly.
        """

        points = deepcopy(points)
        if self._should_normalise:
            points = self._normalize(points)
        if self._should_cluster_only_latency:
            points = self._flatten_logp(points)
        return points

    @staticmethod
    def _flatten_logp(points: list[ExpressionPoint]) -> list[ExpressionPoint]:
        """
        Sets all the logp values in `points` to be equal to 0, in order that they are not used for clustering.
        """
        return [
            ExpressionPoint(
                channel=p.channel,
                latency=p.latency,
                transform=p.transform,
                logp_value=0,
            )
            for p in points
        ]
    
    @staticmethod
    def _normalize(points: list[ExpressionPoint]) -> list[ExpressionPoint]:
        """
        Normalizes points by scaling the latency and magnitude dimensions, depending on whether only latency is
        considered.

        Args:
            points (list[ExpressionPoint]): A list of points to normalize.

        Returns:
            list[ExpressionPoint]: The normalized points.
        """

        # Normalise latency
        latency_sum = sum(p.latency for p in points)
        log_pval_sum = sum(p.logp_value for p in points)

        return [
            ExpressionPoint(
                channel=p.channel,
                latency=p.latency / latency_sum,
                transform=p.transform,
                logp_value=p.logp_value / log_pval_sum)
            for p in points
        ]

    def _get_denoised_time_series(self, points: list[ExpressionPoint]) -> list[ExpressionPoint]:
        """
        For a given list of ExpressionPoint, extract out the most significant spike for each cluster along with its
        associated latency and ignore anomalies.

        Assumes self._clusterer.fit() has been called already.

        :param points: list of points for the time series.
        :returns: list of points (denoised).
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
    def __init__(
            self,
            should_normalise: bool = True,
            should_cluster_only_latency: bool = False,
            should_max_pool: bool = False,
            exclude_logp_vals_above: float | None = None,
            exclude_points_above_n_sigma: float | None = None,
            should_shuffle: bool = True,
            bin_significance_threshold: int = 1,
            bin_size: int = 1,
    ):
        super().__init__(
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            exclude_logp_vals_above,
            exclude_points_above_n_sigma,
            should_shuffle
        )
        self._clusterer = MaxPoolClusterer(bin_significance_threshold, bin_size)


class AdaptiveMaxPoolingStrategy(DenoisingStrategy):
    def __init__(
            self,
            should_normalise: bool = True,
            should_cluster_only_latency: bool = False,
            should_max_pool: bool = False,
            exclude_logp_vals_above: float | None = None,
            exclude_points_above_n_sigma: float | None = None,
            bin_significance_threshold: int = 1,
            base_bin_size: int = 1,
    ):
        super().__init__(
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            exclude_logp_vals_above,
            exclude_points_above_n_sigma,
            should_shuffle=False,  # AMP assumes a sorted time series, so avoid shuffling.
        )                          # Mainly because AMP uses a sliding window algorithm to merge significant clusters.
        self._clusterer = AdaptiveMaxPoolClusterer(bin_significance_threshold, base_bin_size)


class GMMStrategy(DenoisingStrategy):
    def __init__(
            self,
            should_normalise: bool = True,
            should_cluster_only_latency: bool = True,
            should_max_pool: bool = False,
            exclude_logp_vals_above: float | None = None,
            exclude_points_above_n_sigma: float | None = None,
            should_shuffle: bool = True,
            number_of_clusters_upper_bound: int = 2,
            covariance_type: str = "full",
            max_iter: int = 1000,
            n_init: int = 5,
            init_params: str = "kmeans",
            random_state: Optional[int] = None,
            should_evaluate_using_aic: bool = True,
    ):
        super().__init__(
            should_normalise,
            should_cluster_only_latency,
            should_max_pool,
            exclude_logp_vals_above,
            exclude_points_above_n_sigma,
            should_shuffle
        )
        self._clusterer = GMMClusterer(
            number_of_clusters_upper_bound,
            covariance_type,
            max_iter,
            n_init,
            init_params,
            random_state,
            should_evaluate_using_aic,
        )


class DBSCANStrategy(DenoisingStrategy):
    def __init__(
            self,
            should_normalise: bool = False,
            should_cluster_only_latency: bool = False,
            should_max_pool: bool = False,
            exclude_logp_vals_above: float | None = None,
            exclude_points_above_n_sigma: float | None = None,
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
            exclude_logp_vals_above,
            exclude_points_above_n_sigma,
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
            should_normalise: bool = True,
            should_cluster_only_latency: bool = True,
            should_max_pool: bool = False,
            exclude_logp_vals_above: float | None = None,
            exclude_points_above_n_sigma: float | None = None,
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
            exclude_logp_vals_above,
            exclude_points_above_n_sigma,
            should_shuffle
        )
        self._clusterer = MeanShiftClusterer(
            bandwidth=bandwidth,
            seeds=seeds,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            n_jobs=n_jobs,
        )
