from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from math import floor
from typing import List, Dict, Self, Optional

from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import numpy as np

from kymata.ippm.constants import LATENCY_OFFSET


_ANOMALOUS_CLUSTER_TAG = -1


class CustomClusterer(ABC):
    """
    You need to override these methods to create a new clusterer. self.labels assigns each datapoint
    in df to a cluster. Set anomalies to ANOMALOUS_TAG.

    self.labels_ is a list of cluster labels. It has the same size as the dataset.
    """

    def __init__(self):
        self.labels: list[int] = []

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> Self:
        raise NotImplementedError()

    def reset(self):
        self.labels = []


class MaxPoolClusterer(CustomClusterer):
    def __init__(self,
                 label_significance_threshold: int = 15,
                 label_size: int = 25):
        super().__init__()
        self._label_significance_threshold = label_significance_threshold
        self._label_size = label_size

    def fit(self, df: pd.DataFrame) -> Self:
        labels = self._assign_points_to_labels(df)
        count_of_data_per_label = Counter(labels)
        self.labels = self._tag_labels_below_label_significance_threshold_as_anomalies(self.labels, count_of_data_per_label)
        return self

    def _assign_points_to_labels(self, df_with_latency: pd.DataFrame, latency_col_index: int = 0) -> List[int]:
        def __get_bin_index(latency: float) -> int:
            return floor(
                (latency + LATENCY_OFFSET) / self._label_size
            )  # floor because end is exclusive

        return list(map(__get_bin_index, df_with_latency.iloc[:, latency_col_index]))

    def _tag_labels_below_label_significance_threshold_as_anomalies(self, labels: list[int], count_of_data_per_bin: Dict[int, int]):
        for label, count in count_of_data_per_bin.items():
            if count < self._label_significance_threshold:
                labels = MaxPoolClusterer._map_label_to_new_label(label, _ANOMALOUS_CLUSTER_TAG, labels)
        return labels

    @staticmethod
    def _map_label_to_new_label(old_label: int, new_label: int, labels) -> List[int]:
        return list(map(lambda x: new_label if x == old_label else x, labels))


class AdaptiveMaxPoolClusterer(MaxPoolClusterer):
    """
    NOTE: AMP assumes a sorted time-series, so shuffling is not ideal.
    """
    def __init__(self,
                 label_significance_threshold: int = 5,
                 base_label_size: int = 10):
        super().__init__(label_significance_threshold, base_label_size)
        self.labels = []
        self._base_label_size = base_label_size

    def fit(self, df: pd.DataFrame) -> Self:
        labels = self._assign_points_to_labels(df)
        count_of_data_per_label = Counter(labels)
        labels = self._tag_labels_below_label_significance_threshold_as_anomalies(labels, count_of_data_per_label)
        self.labels = self._merge_significant_labels(labels, df)
        return self

    def _merge_significant_labels(self, labels: list[int], df: pd.DataFrame) -> List[int]:
        def __is_insignificant_label(end_index: int) -> bool:
            return labels[end_index] == _ANOMALOUS_CLUSTER_TAG

        def __labels_are_not_adjacent_and_previous_was_significant(end_index: int) -> bool:
            if __is_insignificant_label(end_index - 1):
                # we do not care about the gap between insignificant label and significant label
                return False
            current_latency = df.iloc[end_index, 0]
            previous_latency = df.iloc[end_index - 1, 0]
            return current_latency - previous_latency > self._base_label_size

        def __reached_end(end_index: int) -> bool:
            return end_index == len(labels) - 1

        new_label = 0
        start_index = 0
        for end_index in range(len(labels)):
            if (
                __is_insignificant_label(end_index)
                or __labels_are_not_adjacent_and_previous_was_significant(end_index)
                or __reached_end(end_index)
            ):
                labels = self._merge_labels(
                    labels,
                    start_index,
                    end_index,
                    new_label,
                    __is_insignificant_label(end_index),
                )
                new_label += 1

                if __is_insignificant_label(end_index):
                    start_index = end_index + 1
                else:
                    start_index = end_index

        return labels

    @staticmethod
    def _merge_labels(labels: list[str],
                      start_index: int, end_index: int,
                      new_label: int,
                      current_label_is_anomalous: bool):
        if start_index < end_index:
            if not current_label_is_anomalous:
                # inclusive merge
                end_index += 1

            for index in range(start_index, end_index):
                labels[index] = new_label

        return labels


class GMMClusterer(CustomClusterer):
    def __init__(
        self,
        number_of_clusters_upper_bound: int = 5,
        covariance_type: str = "full",
        max_iter: int = 1000,
        n_init: int = 8,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        should_evaluate_using_AIC: bool = False,
    ):
        """
        Akaike Information Criterion: -2 * log(L) + 2 * k where L = likelihood
                                                                k = # of model parameters
        Bayesian Information Criterion: -2 * log(L) + log(N) * where L and k defined as above
                                                                     N = # of datapoints

        :param number_of_clusters_upper_bound:
        :param covariance_type:
        :param max_iter:
        :param n_init:
        :param init_params:
        :param random_state:
        :param should_evaluate_using_AIC:
        """
        super().__init__()
        self._number_of_clusters_upper_bound = number_of_clusters_upper_bound
        self._covariance_type = covariance_type
        self._max_iter = max_iter
        self._n_init = n_init
        self._init_params = init_params
        self._random_state = random_state
        self._should_evaluate_using_AIC = should_evaluate_using_AIC

    def fit(self, df: pd.DataFrame) -> Self:
        optimal_model = self._grid_search_for_optimal_number_of_clusters(df)
        if optimal_model is not None:
            # None if no data.
            self.labels = optimal_model.predict(df)
            # do not remove anomalies for now
            #self.labels_ = self._tag_low_loglikelihood_points_as_anomalous(
            #    df, optimal_model
            #)
        return self

    @ignore_warnings(category=ConvergenceWarning)
    def _grid_search_for_optimal_number_of_clusters(
        self, df: pd.DataFrame
    ) -> GaussianMixture:
        """
        Quick 101 to model evaluation:
            - Likelihood for a datapoint represents the probability that the datapoint came from the estimated distribution.
              Therefore, the higher the likelihood, the better the fit.
            - Log-Likelihood maps the Likelihood to (-inf, 0]. We still want to maximise this.
            - AIC and BIC use negative Log-Likeliood, so we now attempt to minimise them.
              You can interpret both of these metrics as negative log-likelihood with a model complexity penalty.

        :param df:
        :return:
        """

        def __evaluate_fit(data: pd.DataFrame, fitted_model: GaussianMixture) -> float:
            return (
                fitted_model.aic(data)
                if self._should_evaluate_using_AIC
                else fitted_model.bic(data)
            )

        optimal_penalised_loglikelihood = np.inf
        optimal_model = None
        for number_of_clusters in range(1, self._number_of_clusters_upper_bound):
            if number_of_clusters > len(df) or len(df) == 1:
                self.labels = [0 for _ in range(len(df))]  # default label == 0.
                break

            copy_of_df = deepcopy(df)

            model = GaussianMixture(
                n_components=number_of_clusters,
                covariance_type=self._covariance_type,
                max_iter=self._max_iter,
                n_init=self._n_init,
                init_params=self._init_params,
                random_state=self._random_state,
            )

            max_retries = 3
            for _ in range(max_retries):
                model.fit(copy_of_df)
                covar_matrices = model.covariances_
                if self._all_matrices_invertible(covar_matrices):
                    break

            penalised_loglikelihood = __evaluate_fit(copy_of_df, model)
            if penalised_loglikelihood < optimal_penalised_loglikelihood:
                optimal_penalised_loglikelihood = penalised_loglikelihood
                optimal_model = model

        return optimal_model
    
    @staticmethod
    def _all_matrices_invertible(covar_matrices: np.array) -> bool:
        def __is_not_invertible(num_rows: int, num_cols: int, matrix: np.array) -> bool:
            return (
                num_rows != num_cols or
                num_rows != np.linalg.matrix_rank(matrix)
            )
        for component_covar_matrix in covar_matrices:
            num_rows = component_covar_matrix.shape[0]
            num_cols = component_covar_matrix.shape[1]
            if __is_not_invertible(num_rows, num_cols, component_covar_matrix):
                return False
        return True

    def _tag_low_loglikelihood_points_as_anomalous(
        self,
        df: pd.DataFrame,
        optimal_model: GaussianMixture,
        anomaly_percentile_threshold: int = 5,
    ) -> List[int]:
        def __update_labels_to_anomalous_label(log_likelihoods, anomaly_threshold):
            # we do > cus more negative loglikelihood, the higher the likelihood.
            return list(
                map(
                    lambda x: _ANOMALOUS_CLUSTER_TAG if x[0] < anomaly_threshold else x[1],
                    zip(log_likelihoods, self.labels),
                )
            )

        log_likelihood = optimal_model.score_samples(df)
        threshold = np.percentile(log_likelihood, anomaly_percentile_threshold)
        return __update_labels_to_anomalous_label(log_likelihood, threshold)


class DBSCANClusterer(CustomClusterer):
    def __init__(self,
                 eps: int = 10,
                 min_samples: int = 2,
                 metric: str = "euclidean",
                 algorithm: str = "auto",
                 leaf_size: int = 30,
                 n_jobs: int = -1,
                 metric_params: Optional[dict] = None):
        super().__init__()
        self._dbscan: DBSCAN = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs)

    def fit(self, df: pd.DataFrame) -> Self:
        # A thin wrapper around DBSCAN
        self._dbscan.fit(df)
        self.labels = self._dbscan.labels_
        return self


class MeanShiftClusterer(CustomClusterer):
    def __init__(self,
                 cluster_all: bool = False,
                 bandwidth: float = 30,
                 seeds: Optional[int] = None,
                 min_bin_freq: int = 2,
                 n_jobs: int = -1):
        super().__init__()
        self._meanshift = MeanShift(
            bandwidth=bandwidth,
            seeds=seeds,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            n_jobs=n_jobs)

    def fit(self, df: pd.DataFrame) -> Self:
        # A thin wrapper around MeanShift
        self._meanshift.fit(df)
        self.labels = self._meanshift.labels_
        return self
