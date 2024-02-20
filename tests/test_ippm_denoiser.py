import pytest

from math import isclose

import pandas as pd
import numpy as np

from copy import deepcopy

from kymata.ippm import denoiser
from kymata.ippm.data_tools import IPPMHexel

# Set up shared test hexels
self_test_hexels = {
    'func1' : IPPMHexel('func1'),
    'func2' : IPPMHexel('func2'),
    'func3' : IPPMHexel('func3')
}
self_test_hexels['func1'].left_best_pairings = [
    (20, 1e-66), (35, 1e-50)
]
self_test_hexels['func2'].left_best_pairings = [
    (10, 1e-2), (25, 1e-56), (75, 1e-80)
]
self_test_hexels['func3'].left_best_pairings = [
    (70, 1e-75), (120, 1e-90), (130, 1e-3)
]
self_test_hexels['func1'].right_best_pairings = [
    (23, 1e-75), (35, 1e-44), (66, 1e-50)
]
self_test_hexels['func2'].right_best_pairings = [
    (45, 1e-60), (75, 1e-55), (80, 1e-45)
]
self_test_hexels['func3'].right_best_pairings = [
    (110, 1e-75), (120, 1e-90)
]


def test_Should_MaxPoolerHaveDefaultParams_When_noParams():
    pooler = denoiser.MaxPooler()
    assert pooler._threshold == 15
    assert pooler._bin_sz == 25


def test_Should_maxPoolerHaveCustomParams_When_params():
    custom_threshold = 35
    custom_bin_sz = 10
    pooler = denoiser.MaxPooler(threshold=custom_threshold, bin_sz=custom_bin_sz)
    assert pooler._threshold == custom_threshold
    assert pooler._bin_sz == custom_bin_sz


def test_Should_maxPoolerThrowError_When_invalidParams():
    with pytest.raises(ValueError):
        denoiser.MaxPooler(bin_sz='abc')
    with pytest.raises(ValueError):
        denoiser.MaxPooler(threshold='lmpq')


def test_Should_gmmHaveDefaultParams_When_noParams():
    gmm = denoiser.GMM()
    default_vals = [5, 'full', 1000, 8, 'kmeans', None]
    actual_vals = [gmm._max_gaussians, gmm._covariance_type, gmm._max_iter, gmm._n_init, gmm._init_params,
                   gmm._random_state]
    assert default_vals == actual_vals


def test_Should_gmmHaveCustomParams_When_params():
    custom_max_gaussians = 4
    custom_covar_type = 'spherical'
    custom_max_iter = 150
    custom_n_init = 5
    custom_init_params = 'kmeans'
    custom_random_state = 1
    gmm = denoiser.GMM(max_gaussians=custom_max_gaussians, covariance_type=custom_covar_type,
                       max_iter=custom_max_iter, n_init=custom_n_init,
                       init_params=custom_init_params, random_state=custom_random_state)
    predicted_vals = [custom_max_gaussians, custom_covar_type, custom_max_iter, custom_n_init, custom_init_params,
                      custom_random_state]
    actual_vals = [gmm._max_gaussians, gmm._covariance_type, gmm._max_iter, gmm._n_init, gmm._init_params,
                   gmm._random_state]
    assert predicted_vals == actual_vals


def test_Should_gmmThrowError_When_invalidParams():
    with pytest.raises(ValueError):
        denoiser.GMM(max_gaussians='a')
    with pytest.raises(ValueError):
        denoiser.GMM(covariance_type='b')
    with pytest.raises(ValueError):
        denoiser.GMM(max_iter='c')
    with pytest.raises(ValueError):
        denoiser.GMM(n_init='d')
    with pytest.raises(ValueError):
        denoiser.GMM(init_params='e')
    with pytest.raises(ValueError):
        denoiser.GMM(random_state='f')


def test_Should_dbscanHaveDefaultParams_When_noParams():
    dbscan = denoiser.DBSCAN()
    default_vals = [10, 2, 'euclidean', None, 'auto', 30, -1]
    params = dbscan._clusterer.get_params(deep=False)
    actual_vals = [params['eps'], params['min_samples'], params['metric'], params['metric_params'],
                   params['algorithm'],
                   params['leaf_size'], params['n_jobs']]
    assert default_vals == actual_vals


def test_Should_dbscanHaveCustomParams_When_params():
    custom_eps = 15
    custom_min_samples = 3
    custom_metric = 'cosine'
    custom_metric_params = None
    custom_algorithm = 'kd_tree'
    custom_leaf_size = 15
    custom_n_jobs = -1
    dbscan = denoiser.DBSCAN(eps=custom_eps, min_samples=custom_min_samples, metric=custom_metric,
                             metric_params=custom_metric_params,
                             algorithm=custom_algorithm, leaf_size=custom_leaf_size, n_jobs=custom_n_jobs)
    predicted_vals = [custom_eps, custom_min_samples, custom_metric, custom_metric_params, custom_algorithm,
                      custom_leaf_size, custom_n_jobs]
    params = dbscan._clusterer.get_params(deep=False)
    actual_vals = [params['eps'], params['min_samples'], params['metric'], params['metric_params'],
                   params['algorithm'],
                   params['leaf_size'], params['n_jobs']]
    assert predicted_vals == actual_vals


def test_Should_dbscanThrowError_When_invalidParams():
    with pytest.raises(ValueError):
        denoiser.DBSCAN(eps='a')
    with pytest.raises(ValueError):
        denoiser.DBSCAN(min_samples='b')
    with pytest.raises(ValueError):
        denoiser.DBSCAN(metric=1)
    with pytest.raises(ValueError):
        denoiser.DBSCAN(metric_params='d')
    with pytest.raises(ValueError):
        denoiser.DBSCAN(algorithm='e')
    with pytest.raises(ValueError):
        denoiser.DBSCAN(leaf_size='f')
    with pytest.raises(ValueError):
        denoiser.DBSCAN(n_jobs='g')


def test_Should_meanShiftHaveDefaultParams_When_noParams():
    mean_shift = denoiser.MeanShift()
    params = mean_shift._clusterer.get_params(deep=False)
    default_vals = [False, 30, None, 2, -1]
    actual_vals = [params['cluster_all'], params['bandwidth'], params['seeds'], params['min_bin_freq'],
                   params['n_jobs']]
    assert default_vals == actual_vals


def test_Should_meanShiftHaveCustomParams_When_params():
    custom_cluster_all = True
    custom_bandwidth = 2
    custom_seeds = None
    custom_min_bin_freq = 3
    custom_n_jobs = -1
    mean_shift = denoiser.MeanShift(cluster_all=custom_cluster_all, bandwidth=custom_bandwidth, seeds=custom_seeds,
                                    min_bin_freq=custom_min_bin_freq, n_jobs=custom_n_jobs)
    params = mean_shift._clusterer.get_params(deep=False)
    predicted_vals = [custom_cluster_all, custom_bandwidth, custom_seeds, custom_min_bin_freq, custom_n_jobs]
    actual_vals = [params['cluster_all'], params['bandwidth'], params['seeds'], params['min_bin_freq'],
                   params['n_jobs']]
    assert predicted_vals == actual_vals


def test_Should_meanShiftThrowException_When_invalidParams():
    with pytest.raises(ValueError):
        denoiser.MeanShift(cluster_all='a')
    with pytest.raises(ValueError):
        denoiser.MeanShift(bandwidth='b')
    with pytest.raises(ValueError):
        denoiser.MeanShift(seeds=1)
    with pytest.raises(ValueError):
        denoiser.MeanShift(min_bin_freq='d')
    with pytest.raises(ValueError):
        denoiser.MeanShift(n_jobs='e')


def test_Should_checkHemiReturn_When_validParams():
    clusterer = denoiser.DenoisingStrategy()
    try:
        clusterer._check_hemi('rightHemisphere')
        clusterer._check_hemi('leftHemisphere')
    except ValueError:
        pytest.fail('check_hemi failed for valid params.')


def test_Should_checkHemiThrowException_When_invalidParams():
    clusterer = denoiser.DenoisingStrategy()
    with pytest.raises(ValueError):
        clusterer._check_hemi('right hemisphere')

def test_Should_clusterBin_When_validInput():
    df = pd.DataFrame(columns=['Latency', 'Mag'])
    df.loc[len(df)] = [205, 1e-10]
    df.loc[len(df)] = [210, 1e-15]
    df.loc[len(df)] = [215, 1e-9]
    df.loc[len(df)] = [220, 1e-20]

    denoiser_ = denoiser.MaxPooler(threshold=3)  # bin_sz = 25 by default, threshold = 3, so we dont need large df.
    bin_min, lat_min, num_seen, r_idx = denoiser_._cluster_bin(df, r_idx=0, latency=200)
    assert bin_min == 1e-20
    assert lat_min == 220
    assert num_seen == 4
    assert r_idx == 4

def test_Should_clusterBinTerminate_When_outOfLatencyBin():
    df = pd.DataFrame(columns=['Latency', 'Mag'])
    df.loc[len(df)] = [205, 1e-10]
    df.loc[len(df)] = [210, 1e-15]
    df.loc[len(df)] = [215, 1e-9]
    df.loc[len(df)] = [250, 1e-20]

    denoiser_ = denoiser.MaxPooler(threshold=3)  # bin_sz = 25, threshold = 15 by default
    bin_min, lat_min, num_seen, r_idx = denoiser_._cluster_bin(df, r_idx=0, latency=200)
    assert bin_min == 1e-15
    assert lat_min == 210  # 1e-20 is out of latency bin, so it should terminate early
    assert num_seen == 3
    assert r_idx == 3

def test_Should_clusterBinsExit_When_emptyBin():
    df = pd.DataFrame(columns=['Latency', 'Mag'])
    df.loc[len(df)] = [205, 1e-10]
    df.loc[len(df)] = [210, 1e-15]
    df.loc[len(df)] = [215, 1e-9]
    df.loc[len(df)] = [250, 1e-20]

    denoiser_ = denoiser.MaxPooler(threshold=3)  # bin_sz = 25, threshold = 15 by default
    bin_min, lat_min, num_seen, r_idx = denoiser_._cluster_bin(df, r_idx=4, latency=275)
    assert bin_min == float('inf')
    assert lat_min is None  # 1e-20 is out of latency bin, so it should terminate early
    assert num_seen == 0
    assert r_idx == 4

def test_Should_estimateAlpha():
    alpha = denoiser.DenoisingStrategy()._estimate_alpha()
    predicted_alpha = 3.55e-15
    assert isclose(alpha, predicted_alpha, abs_tol=1e-15)

def test_Should_hexelsToDf_When_validInput():
    test_hexels = {'f1': IPPMHexel('f1'), 'f2': IPPMHexel('f2')}
    test_hexels['f1'].right_best_pairings = [(10, 1e-25), (25, 1e-23)]
    test_hexels['f2'].right_best_pairings = [(50, 1e-10), (55, 1e-44)]
    denoiser_ = denoiser.DenoisingStrategy()
    fs = ['f1', 'f2']
    i = 0
    for func, df in denoiser_._hexels_to_df(test_hexels, 'rightHemisphere'):
        assert fs[i] == func
        if fs[i] == 'f2':
            assert test_hexels['f2'].right_best_pairings[1][0] == df.iloc[0, 0]
            assert test_hexels['f2'].right_best_pairings[1][1] == df.iloc[0, 1]
        else:
            assert test_hexels['f1'].right_best_pairings[0][0] == df.iloc[0, 0]
            assert test_hexels['f1'].right_best_pairings[0][1] == df.iloc[0, 1]
            assert test_hexels['f1'].right_best_pairings[1][0] == df.iloc[1, 0]
            assert test_hexels['f1'].right_best_pairings[1][1] == df.iloc[1, 1]
        i += 1

def test_Should_hexelsToDf_When_leftHemisphere():
    test_hexels = {'f1': IPPMHexel('f1'), 'f2': IPPMHexel('f2')}
    test_hexels['f1'].left_best_pairings = [(10, 1e-25), (25, 1e-23)]
    test_hexels['f2'].left_best_pairings = [(50, 1e-10), (55, 1e-44)]
    denoiser_ = denoiser.DenoisingStrategy()
    fs = ['f1', 'f2']
    i = 0
    for func, df in denoiser_._hexels_to_df(test_hexels, 'leftHemisphere'):
        assert fs[i] == func
        if fs[i] == 'f2':
            assert test_hexels['f2'].left_best_pairings[1][0] == df.iloc[0, 0]
            assert test_hexels['f2'].left_best_pairings[1][1] == df.iloc[0, 1]
        else:
            assert test_hexels['f1'].left_best_pairings[0][0] == df.iloc[0, 0]
            assert test_hexels['f1'].left_best_pairings[0][1] == df.iloc[0, 1]
            assert test_hexels['f1'].left_best_pairings[1][0] == df.iloc[1, 0]
            assert test_hexels['f1'].left_best_pairings[1][1] == df.iloc[1, 1]
        i += 1

def test_Should_hexelsToDf_When_emptyInput():
    test_hexels = {'f1': IPPMHexel('f1'), 'f2': IPPMHexel('f2')}
    denoiser_ = denoiser.DenoisingStrategy()
    for _, df in denoiser_._hexels_to_df(test_hexels, 'leftHemisphere'):
        assert len(df) == 0

def test_Should_updatePairings_When_validInput():
    test_hexels = {'f1': IPPMHexel('f1'), 'f2': IPPMHexel('f2')}
    test_hexels['f1'].right_best_pairings = [(10, 1e-20), (15, 1e-2)]
    test_hexels['f2'].right_best_pairings = [(200, 1e-1), (250, 1e-50)]
    denoised = [(10, 1e-20)]
    denoiser_ = denoiser.DenoisingStrategy()
    hexels = denoiser_._update_pairings(test_hexels, 'f1', denoised, 'rightHemisphere')
    assert hexels['f1'].right_best_pairings == denoised
    assert hexels['f2'].right_best_pairings == [(200, 1e-1), (250, 1e-50)]

def test_Should_updatePairings_When_leftHemisphere():
    test_hexels = {'f1': IPPMHexel('f1'), 'f2': IPPMHexel('f2')}
    test_hexels['f1'].left_best_pairings = [(10, 1e-20), (15, 1e-2)]
    test_hexels['f2'].left_best_pairings = [(200, 1e-1), (250, 1e-50)]
    denoised = [(10, 1e-20)]
    denoiser_ = denoiser.DenoisingStrategy()
    hexels = denoiser_._update_pairings(test_hexels, 'f1', denoised, 'leftHemisphere')
    assert hexels['f1'].left_best_pairings == denoised
    assert hexels['f2'].left_best_pairings == [(200, 1e-1), (250, 1e-50)]

def test_Should_filterSpikes_When_validInput():
    clusterer = denoiser.DenoisingStrategy()
    df = pd.DataFrame(columns=['Latency', 'Mag'])
    self_test_hexels2 = deepcopy(self_test_hexels)
    df = clusterer._filter_spikes(self_test_hexels2['func2'].left_best_pairings, df, alpha=3e-15)
    assert df.iloc[0, 0] == 25
    assert df.iloc[0, 1] == 1e-56
    assert df.iloc[1, 0] == 75
    assert df.iloc[1, 1] == 1e-80

def test_Should_filterSpikes_When_emptyInput():
    self_test_hexels2 = deepcopy(self_test_hexels)
    self_test_hexels2['func1'].right_best_pairings = []
    clusterer = denoiser.DenoisingStrategy()
    df = pd.DataFrame(columns=['Latency', 'Mag'])
    df = clusterer._filter_spikes(self_test_hexels2['func1'].right_best_pairings, df, alpha=3e-15)
    assert len(df) == 0

def test_Should_filterSpikes_When_noSignificantSpikes():
    self_test_hexels2 = deepcopy(self_test_hexels)
    clusterer = denoiser.DenoisingStrategy()
    df = pd.DataFrame(columns=['Latency', 'Mag'])
    # increase alpha so all are insignificant.
    df = clusterer._filter_spikes(self_test_hexels2['func2'].left_best_pairings, df, alpha=3e-100)
    assert len(df) == 0

def test_Should_maxPoolerCluster_When_validInput():
    pooler = denoiser.MaxPooler(bin_sz=50, threshold=1)
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = pooler.cluster(self_test_hexels2, 'rightHemisphere')
    assert denoised['func1'].right_best_pairings == [(23, 1e-75), (66, 1e-50)]
    assert denoised['func2'].right_best_pairings == [(45, 1e-60), (75, 1e-55)]
    assert denoised['func3'].right_best_pairings == [(120, 1e-90)]

def test_Should_maxPoolerCluster_When_noSignificantBins():
    # all bins < threshold for significance.
    pooler = denoiser.MaxPooler(bin_sz=50)
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = pooler.cluster(self_test_hexels2, 'rightHemisphere')
    assert denoised['func1'].right_best_pairings == []
    assert denoised['func2'].right_best_pairings == []
    assert denoised['func3'].right_best_pairings == []

def test_Should_adaptiveMaxPoolerCluster_When_validInputRightHemisphere():
    clusterer = denoiser.AdaptiveMaxPooler(base_bin_sz=50, threshold=2)
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'rightHemisphere')

    assert denoised['func1'].right_best_pairings == [(23, 1e-75)]
    assert denoised['func2'].right_best_pairings == [(75, 1e-55)]
    assert denoised['func3'].right_best_pairings == [(120, 1e-90)]

def test_Should_adaptiveMaxPoolerCluster_When_validInputLeftHemisphere():
    clusterer = denoiser.AdaptiveMaxPooler(base_bin_sz=50, threshold=2)
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'leftHemisphere')

    assert denoised['func1'].left_best_pairings == [(20, 1e-66)]
    assert denoised['func2'].left_best_pairings == []
    assert denoised['func3'].left_best_pairings == []


def test_Should_gmmCluster_When_validInputRightHemisphere():
    """
        What we expect to happen: Each data point will be it's own cluster, except points that are identified as insignificant.
        Why: The likelihood is maximised by assigning each point to it's own cluster. It is the side-effect
             of having too little data points. 
    """
    np.random.seed(0)  # for reproducibility of ML algorithms
    clusterer = denoiser.GMM()
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'rightHemisphere')
    f2_expected = self_test_hexels2['func2'].right_best_pairings
    assert denoised['func1'].right_best_pairings == self_test_hexels2['func1'].right_best_pairings
    assert denoised['func2'].right_best_pairings == f2_expected
    assert denoised['func3'].right_best_pairings == self_test_hexels2['func3'].right_best_pairings

def test_Should_gmmCluster_When_validInputLeftHemisphere():
    """
        What we expect to happen: Each data point will be it's own cluster, except points that are identified as insignificant.
        Why: The likelihood is maximised by assigning each point to it's own cluster. It is the side-effect
             of having too little data points. 

        E.g., (10, 0.01) is identified as insignificant since it is greater than alpha. Hence, exclude.
    """
    np.random.seed(0)  # for reproducibility of ML algorithms
    clusterer = denoiser.GMM()
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'leftHemisphere')
    f2_expected = self_test_hexels2['func2'].left_best_pairings
    f3_expected = self_test_hexels2['func3'].left_best_pairings
    f2_expected.remove((10, 0.01))
    f3_expected.remove((130, 0.001))
    assert denoised['func1'].left_best_pairings == self_test_hexels2['func1'].left_best_pairings
    assert denoised['func2'].left_best_pairings == self_test_hexels2['func2'].left_best_pairings
    assert denoised['func3'].left_best_pairings == self_test_hexels2['func3'].left_best_pairings

def test_Should_dbscanCluster_When_validInputRightHemi():
    np.random.seed(0)
    clusterer = denoiser.DBSCAN()
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'rightHemisphere')
    assert [] == denoised['func1'].right_best_pairings
    assert [(75, 1e-55)] == denoised['func2'].right_best_pairings
    assert [(120, 1e-90)] == denoised['func3'].right_best_pairings

def test_Should_dbscanCluster_When_validInputLeftHemi():
    """
        The reason DBSCAN returns empty is that a point is only considered significant if it has at least
        one more point within a 10 ms radius. In this case, none do. Moreover, the magnitudes that are higher than
        alpha are excluded as insignificant.
    """
    np.random.seed(0)
    clusterer = denoiser.DBSCAN()
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'leftHemisphere')
    assert [] == denoised['func1'].left_best_pairings
    assert [] == denoised['func2'].left_best_pairings
    assert [] == denoised['func3'].left_best_pairings

def test_Should_meanShiftCluster_When_validInputRightHemi():
    np.random.seed(0)
    clusterer = denoiser.MeanShift()
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'rightHemisphere')
    assert [(23, 1e-75), (66, 1e-50)] == denoised['func1'].right_best_pairings
    assert [(45.0, 1e-60)] == denoised['func2'].right_best_pairings
    assert [(120.0, 1e-90)] == denoised['func3'].right_best_pairings

def test_Should_meanShiftCluster_When_validInputLeftHemi():
    np.random.seed(0)
    clusterer = denoiser.MeanShift()
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'leftHemisphere')
    f2_expected = self_test_hexels['func2'].left_best_pairings
    f3_expected = self_test_hexels['func3'].left_best_pairings
    f2_expected.remove((10, 0.01))
    f3_expected.remove((130, 0.001))
    assert [(20.0, 1e-66)] == denoised['func1'].left_best_pairings
    assert f2_expected == denoised['func2'].left_best_pairings
    assert self_test_hexels['func3'].left_best_pairings == denoised['func3'].left_best_pairings

def test_Should_getLatencyDim_When_validDfRightHemisphere():
    clusterer = denoiser.DenoisingStrategy()
    self_test_hexels2 = deepcopy(self_test_hexels)
    latency_dfs = []
    for func, df in clusterer._hexels_to_df(self_test_hexels2, 'rightHemisphere'):
        latency_dfs.append(clusterer._get_latency_dim(df))
    
    assert [23, 35, 66] == list(latency_dfs[0].flatten())
    assert [45, 75, 80] == list(latency_dfs[1].flatten())
    assert [110, 120] == list(latency_dfs[2].flatten())

def test_Should_posteriorPool_When_validHexelsRightHemisphere():
    clusterer = denoiser.DenoisingStrategy()
    self_test_hexels2 = deepcopy(self_test_hexels)
    pooled_hexels = clusterer._posterior_pooling(self_test_hexels2, 'rightHemisphere')
    pooled = []
    for func in pooled_hexels.keys():
        pooled.append(pooled_hexels[func].right_best_pairings)

    assert [(23, 1e-75)] == pooled[0]
    assert [(45, 1e-60)] == pooled[1]
    assert [(120, 1e-90)] == pooled[2]

def test_Should_posteriorPool_When_validHexelsLeftHemisphere():
    clusterer = denoiser.DenoisingStrategy()
    self_test_hexels2 = deepcopy(self_test_hexels)
    pooled_hexels = clusterer._posterior_pooling(self_test_hexels2, 'leftHemisphere')
    pooled = []
    for func in pooled_hexels.keys():
        pooled.append(pooled_hexels[func].left_best_pairings)

    assert [(20, 1e-66)] == pooled[0]
    assert [(75, 1e-80)] == pooled[1]
    assert [(120, 1e-90)] == pooled[2]

def test_Should_posteriorPool_When_emptyHexels():
    clusterer = denoiser.DenoisingStrategy()
    self_test_hexels2 = deepcopy(self_test_hexels)
    self_test_hexels2['func1'].right_best_pairings = []
    pooled_hexels = clusterer._posterior_pooling(self_test_hexels2, 'rightHemisphere')
    assert [] == pooled_hexels['func1'].right_best_pairings

def test_Should_maxPoolerPool_When_normalised():
    pooler = denoiser.MaxPooler(bin_sz=50, threshold=1)
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = pooler.cluster(self_test_hexels2, 'rightHemisphere', normalise=True)
    assert denoised['func1'].right_best_pairings == [(23, 1e-75), (66, 1e-50)]
    assert denoised['func2'].right_best_pairings == [(45, 1e-60), (75, 1e-55)]
    assert denoised['func3'].right_best_pairings == [(120, 1e-90)]

def test_Should_adaptiveMaxPoolerPool_When_normalised():
    clusterer = denoiser.AdaptiveMaxPooler(base_bin_sz=50, threshold=2)
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'rightHemisphere', normalise=True)

    assert denoised['func1'].right_best_pairings == [(23, 1e-75)]
    assert denoised['func2'].right_best_pairings == [(75, 1e-55)]
    assert denoised['func3'].right_best_pairings == [(120, 1e-90)]

def test_Should_gmmPool_When_normalised():
    np.random.seed(0) 
    clusterer = denoiser.GMM()
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'rightHemisphere', normalise=True)
    assert denoised['func1'].right_best_pairings == [(23, 1e-75)]
    assert denoised['func2'].right_best_pairings == [(45, 1e-60)]
    assert denoised['func3'].right_best_pairings == [(120, 1e-90)]

def test_Should_dbscanPool_When_normalised():
    np.random.seed(0)
    clusterer = denoiser.DBSCAN()
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'rightHemisphere', normalise=True)
    assert [(23, 1e-75)] == denoised['func1'].right_best_pairings
    assert [(45, 1e-60)] == denoised['func2'].right_best_pairings
    assert [(120, 1e-90)] == denoised['func3'].right_best_pairings

def test_Should_meanShiftPool_When_normalised():
    np.random.seed(0)
    clusterer = denoiser.MeanShift()
    self_test_hexels2 = deepcopy(self_test_hexels)
    denoised = clusterer.cluster(self_test_hexels2, 'rightHemisphere', normalise=True)
    assert [(23, 1e-75)] == denoised['func1'].right_best_pairings
    assert [(45.0, 1e-60)] == denoised['func2'].right_best_pairings
    assert [(120.0, 1e-90)] == denoised['func3'].right_best_pairings
    
